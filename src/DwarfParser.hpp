//===--------------------------- DwarfParser.hpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
//  Parses DWARF CFIs (FDEs and CIEs).
//
//===----------------------------------------------------------------------===//

#ifndef __DWARF_PARSER_HPP__
#define __DWARF_PARSER_HPP__

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

#include "libunwind.h"
#include "dwarf2.h"
#include "Registers.hpp"

#include "config.h"

namespace libunwind {

enum {
  kMaxRegisterNumber = _LIBUNWIND_HIGHEST_DWARF_REGISTER
};

enum RegisterSavedWhere {
  kRegisterUnused,
  kRegisterInCFA,
  kRegisterOffsetFromCFA,
  kRegisterInRegister,
  kRegisterAtExpression,
  kRegisterIsExpression
};

struct RegisterLocation {
  RegisterSavedWhere location;
  int64_t value;
};

/// Information about a frame layout and registers saved determined
/// by "running" the DWARF FDE "instructions"
struct PrologInfo {
  uint32_t          cfaRegister;
  int32_t           cfaRegisterOffset;  // CFA = (cfaRegister)+cfaRegisterOffset
  int64_t           cfaExpression;      // CFA = expression
  uint32_t          spExtraArgSize;
  uint32_t          codeOffsetAtStackDecrement;
  bool              registersInOtherRegisters;
  bool              sameValueUsed;
  RegisterLocation  savedRegisters[kMaxRegisterNumber + 1];
};

struct PrologInfoStackEntry {
  PrologInfoStackEntry(PrologInfoStackEntry *n, const PrologInfo &i)
      : next(n), info(i) {}
  PrologInfoStackEntry *next;
  PrologInfo info;
};

/// Here we use intrinsics because g++ cannot use stdatomic.h, and using c++
/// atomics is not possible. Wrappers are just in case other compiler has
/// different intrinsics.

template <typename Ptr>
static bool atomic_compare_exchange_wrapper(Ptr * target, Ptr * compare, Ptr exchange)
{
    return __atomic_compare_exchange_n(target, compare, exchange, true, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}

static size_t atomic_fetch_add_wrapper(size_t * ptr, size_t val) { /// NOLINT
  return __atomic_fetch_add(ptr, val, __ATOMIC_SEQ_CST);
}

static void *atomic_load_wrapper(void *&ptr) {
  return __atomic_load_n(&ptr, __ATOMIC_SEQ_CST);
}

static void atomic_store_wrapper(void *&ptr, void *val) {
  __atomic_store_n(&ptr, val, __ATOMIC_SEQ_CST);
}


/** Lock free stack from LFAlloc. Author Andrei Gulin.
  * Supports push and pop methods.
  *
  *  Node interface:
  *  struct Node
  *  {
  *      Node * next;
  *  }
  */
template <typename Node>
struct LFStack
{
public:
    Node * pop()
    {
        size_t move_pending_to_head_counter_before = atomic_fetch_add_wrapper(&move_pending_to_head_counter, 0);
        Node * pending_before = pending;

        if (atomic_fetch_add_wrapper(&pop_count, 1) == 0)
        {
            /// If there are no other allocations in progress then try to move pending list into head

            // If (move_pending_to_head_counter_before == move_pending_to_head_counter) then pending list
            // was not  freed by other threads. Hence pending list is not used in any concurrent pop
            // we can move it to head.

            if (pending_before &&
                move_pending_to_head_counter_before == atomic_fetch_add_wrapper(&move_pending_to_head_counter, 0) &&
                atomic_compare_exchange_wrapper(&pending, &pending_before, static_cast<Node *>(nullptr)))
            {
                /// Pick first node from pending list and return it
                Node * result = pending_before;
                pending_before = pending_before->next;

                movePendingListIntoHead(pending_before);
                atomic_fetch_add_wrapper(&move_pending_to_head_counter, 1);
                atomic_fetch_add_wrapper(&pop_count, -1);

                return result;
            }
        }

        auto * ptr = popImpl();
        atomic_fetch_add_wrapper(&pop_count, -1);

        return ptr;
    }

    void push(Node * node)
    {
        /// If there is pop in progress move node to pending list
        if (atomic_fetch_add_wrapper(&pop_count, 0) == 0)
            enqueue(&head, node);
        else
            enqueue(&pending, node);
     }

private:
    Node * popImpl()
    {
        Node * prev_head = head;

        while (prev_head)
        {
            Node * prev_head_next = prev_head->next;

            if (atomic_compare_exchange_wrapper(&head, &prev_head, prev_head_next))
                break;
        }

        return prev_head;
    }

    static void enqueue(Node ** list_head, Node * node)
    {
        while (true)
        {
            Node * prev_head = *list_head;
            node->next = prev_head;

            if (atomic_compare_exchange_wrapper(list_head, &prev_head, node))
                break;
        }
    }

    void movePendingListIntoHead(Node * pending_list_head)
    {
        if (!pending_list_head)
            return;

        Node * pending_list_tail = pending_list_head;
        while (pending_list_tail->next)
            pending_list_tail = pending_list_tail->next;

        while (true)
        {
            Node * prev_head = head;
            pending_list_tail->next = prev_head;

            if (atomic_compare_exchange_wrapper(&head, &prev_head, pending_list_head))
                break;
        }
    }

    Node * head = nullptr;
    Node * pending = nullptr;
    size_t pop_count = 0;
    size_t move_pending_to_head_counter = 0;
};

struct FreeListEntry
{
  FreeListEntry * next = nullptr;
  char buffer[sizeof(libunwind::PrologInfoStackEntry)];
};

/// DWARF instructions DW_CFA_remember_state and DW_CFA_restore_state require stack.
/// libunwind cannot use malloc to be signal safe.
/// It also should not use mmap for every allocation as it is too slow.
/// It can use pre-allocated buffer on stack, but it cannot be large (for coroutines) and cannot be small (it will be not enough).
/// It also cannot use thread-local preallocated buffer because of coroutines.
/// It can use pre-allocated global buffer, but thread synchronization is required.
/// It cannot use mutex because of signal safety.
/// And it cannot use standard C++ library headers because of libunwind is used to build C++.
/// So, we use atomic operations over mmaped pre-allocated global buffer.
class StackBuffer {
private:
  static constexpr size_t entrySize = sizeof(FreeListEntry);
  static constexpr size_t buffer_size = LIBUNWIND_MAX_STACK_SIZE * entrySize;
  static void *buffer;
  static size_t next_not_allocated_entry;
  static void *next_allocated_entry;
  static LFStack<FreeListEntry> stack;

public:
  static FreeListEntry *alloc() {
    /// At first, try to get element from stack.
    {
      FreeListEntry *entry = stack.pop();
      if (entry)
        return static_cast<FreeListEntry *>(entry);
    }

    size_t prev = atomic_fetch_add_wrapper(&next_not_allocated_entry, 1);

    if (prev == 0) {
      /// I am the one who mmaps memory.
      void *res = mmap(nullptr, buffer_size, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

      if (MAP_FAILED == res)
        abort();

      atomic_store_wrapper(buffer, res);
    }

    if ((prev + 1) * entrySize >= buffer_size)
      abort();

    /// Life loop. In case memory was not allocated yet.
    void *data = nullptr;
    while (!data)
      data = atomic_load_wrapper(buffer);

    return reinterpret_cast<FreeListEntry *>(static_cast<FreeListEntry *>(data) + prev);
  }

  static void free(FreeListEntry *entry)
  {
    stack.push(entry);
  }
};

class StackGuard {
public:

  void *push() {
    FreeListEntry *entry = StackBuffer::alloc();
    entry->next = stack_top;
    stack_top = entry;
    return stack_top->buffer;
  }

  void pop() {
    if (stack_top == nullptr)
      abort();

    FreeListEntry *entry = stack_top;
    stack_top = stack_top->next;

    StackBuffer::free(entry);
  }

  ~StackGuard() {
    while (stack_top)
      pop();
  }

private:
    FreeListEntry * stack_top = nullptr;
};
}

namespace libunwind {
/// CFI_Parser does basic parsing of a CFI (Call Frame Information) records.
/// See DWARF Spec for details:
///    http://refspecs.linuxbase.org/LSB_3.1.0/LSB-Core-generic/LSB-Core-generic/ehframechpt.html
///
template <typename A>
class CFI_Parser {
public:
  typedef typename A::pint_t pint_t;

  /// Information encoded in a CIE (Common Information Entry)
  struct CIE_Info {
    pint_t    cieStart;
    pint_t    cieLength;
    pint_t    cieInstructions;
    uint8_t   pointerEncoding;
    uint8_t   lsdaEncoding;
    uint8_t   personalityEncoding;
    uint8_t   personalityOffsetInCIE;
    pint_t    personality;
    uint32_t  codeAlignFactor;
    int       dataAlignFactor;
    bool      isSignalFrame;
    bool      fdesHaveAugmentationData;
    uint8_t   returnAddressRegister;
#if defined(_LIBUNWIND_TARGET_AARCH64)
    bool      addressesSignedWithBKey;
#endif
  };

  /// Information about an FDE (Frame Description Entry)
  struct FDE_Info {
    pint_t  fdeStart;
    pint_t  fdeLength;
    pint_t  fdeInstructions;
    pint_t  pcStart;
    pint_t  pcEnd;
    pint_t  lsda;
  };

  static bool findFDE(A &addressSpace, pint_t pc, pint_t ehSectionStart,
                      uintptr_t sectionLength, pint_t fdeHint, FDE_Info *fdeInfo,
                      CIE_Info *cieInfo);
  static const char *decodeFDE(A &addressSpace, pint_t fdeStart,
                               FDE_Info *fdeInfo, CIE_Info *cieInfo);
  static bool parseFDEInstructions(A &addressSpace, const FDE_Info &fdeInfo,
                                   const CIE_Info &cieInfo, pint_t upToPC,
                                   int arch, PrologInfo *results);

  static const char *parseCIE(A &addressSpace, pint_t cie, CIE_Info *cieInfo);

private:
  static bool parseInstructions(A &addressSpace, pint_t instructions,
                                pint_t instructionsEnd, const CIE_Info &cieInfo,
                                pint_t pcoffset,
                                PrologInfoStackEntry *&rememberStack, int arch,
                                PrologInfo *results,
                                StackGuard &stack);

};


/// Parse a FDE into a CIE_Info and an FDE_Info
template <typename A>
const char *CFI_Parser<A>::decodeFDE(A &addressSpace, pint_t fdeStart,
                                     FDE_Info *fdeInfo, CIE_Info *cieInfo) {
  pint_t p = fdeStart;
  pint_t cfiLength = (pint_t)addressSpace.get32(p);
  p += 4;
  if (cfiLength == 0xffffffff) {
    // 0xffffffff means length is really next 8 bytes
    cfiLength = (pint_t)addressSpace.get64(p);
    p += 8;
  }
  if (cfiLength == 0)
    return "FDE has zero length"; // zero terminator
  uint32_t ciePointer = addressSpace.get32(p);
  if (ciePointer == 0)
    return "FDE is really a CIE"; // this is a CIE not an FDE
  pint_t nextCFI = p + cfiLength;
  pint_t cieStart = p - ciePointer;
  const char *err = parseCIE(addressSpace, cieStart, cieInfo);
  if (err != NULL)
    return err;
  p += 4;
  // Parse pc begin and range.
  pint_t pcStart =
      addressSpace.getEncodedP(p, nextCFI, cieInfo->pointerEncoding);
  pint_t pcRange =
      addressSpace.getEncodedP(p, nextCFI, cieInfo->pointerEncoding & 0x0F);
  // Parse rest of info.
  fdeInfo->lsda = 0;
  // Check for augmentation length.
  if (cieInfo->fdesHaveAugmentationData) {
    pint_t augLen = (pint_t)addressSpace.getULEB128(p, nextCFI);
    pint_t endOfAug = p + augLen;
    if (cieInfo->lsdaEncoding != DW_EH_PE_omit) {
      // Peek at value (without indirection).  Zero means no LSDA.
      pint_t lsdaStart = p;
      if (addressSpace.getEncodedP(p, nextCFI, cieInfo->lsdaEncoding & 0x0F) !=
          0) {
        // Reset pointer and re-parse LSDA address.
        p = lsdaStart;
        fdeInfo->lsda =
            addressSpace.getEncodedP(p, nextCFI, cieInfo->lsdaEncoding);
      }
    }
    p = endOfAug;
  }
  fdeInfo->fdeStart = fdeStart;
  fdeInfo->fdeLength = nextCFI - fdeStart;
  fdeInfo->fdeInstructions = p;
  fdeInfo->pcStart = pcStart;
  fdeInfo->pcEnd = pcStart + pcRange;
  return NULL; // success
}

/// Scan an eh_frame section to find an FDE for a pc
template <typename A>
bool CFI_Parser<A>::findFDE(A &addressSpace, pint_t pc, pint_t ehSectionStart,
                            uintptr_t sectionLength, pint_t fdeHint,
                            FDE_Info *fdeInfo, CIE_Info *cieInfo) {
  //fprintf(stderr, "findFDE(0x%llX)\n", (long long)pc);
  pint_t p = (fdeHint != 0) ? fdeHint : ehSectionStart;
  const pint_t ehSectionEnd = (sectionLength == UINTPTR_MAX)
                                  ? static_cast<pint_t>(-1)
                                  : (ehSectionStart + sectionLength);
  while (p < ehSectionEnd) {
    pint_t currentCFI = p;
    //fprintf(stderr, "findFDE() CFI at 0x%llX\n", (long long)p);
    pint_t cfiLength = addressSpace.get32(p);
    p += 4;
    if (cfiLength == 0xffffffff) {
      // 0xffffffff means length is really next 8 bytes
      cfiLength = (pint_t)addressSpace.get64(p);
      p += 8;
    }
    if (cfiLength == 0)
      return false; // zero terminator
    uint32_t id = addressSpace.get32(p);
    if (id == 0) {
      // Skip over CIEs.
      p += cfiLength;
    } else {
      // Process FDE to see if it covers pc.
      pint_t nextCFI = p + cfiLength;
      uint32_t ciePointer = addressSpace.get32(p);
      pint_t cieStart = p - ciePointer;
      // Validate pointer to CIE is within section.
      if ((ehSectionStart <= cieStart) && (cieStart < ehSectionEnd)) {
        if (parseCIE(addressSpace, cieStart, cieInfo) == NULL) {
          p += 4;
          // Parse pc begin and range.
          pint_t pcStart =
              addressSpace.getEncodedP(p, nextCFI, cieInfo->pointerEncoding);
          pint_t pcRange = addressSpace.getEncodedP(
              p, nextCFI, cieInfo->pointerEncoding & 0x0F);
          // Test if pc is within the function this FDE covers.
          if ((pcStart < pc) && (pc <= pcStart + pcRange)) {
            // parse rest of info
            fdeInfo->lsda = 0;
            // check for augmentation length
            if (cieInfo->fdesHaveAugmentationData) {
              pint_t augLen = (pint_t)addressSpace.getULEB128(p, nextCFI);
              pint_t endOfAug = p + augLen;
              if (cieInfo->lsdaEncoding != DW_EH_PE_omit) {
                // Peek at value (without indirection).  Zero means no LSDA.
                pint_t lsdaStart = p;
                if (addressSpace.getEncodedP(
                        p, nextCFI, cieInfo->lsdaEncoding & 0x0F) != 0) {
                  // Reset pointer and re-parse LSDA address.
                  p = lsdaStart;
                  fdeInfo->lsda = addressSpace
                      .getEncodedP(p, nextCFI, cieInfo->lsdaEncoding);
                }
              }
              p = endOfAug;
            }
            fdeInfo->fdeStart = currentCFI;
            fdeInfo->fdeLength = nextCFI - currentCFI;
            fdeInfo->fdeInstructions = p;
            fdeInfo->pcStart = pcStart;
            fdeInfo->pcEnd = pcStart + pcRange;
            return true;
          } else {
            // pc is not in begin/range, skip this FDE
          }
        } else {
          // Malformed CIE, now augmentation describing pc range encoding.
        }
      } else {
        // malformed FDE.  CIE is bad
      }
      p = nextCFI;
    }
  }
  return false;
}

/// Extract info from a CIE
template <typename A>
const char *CFI_Parser<A>::parseCIE(A &addressSpace, pint_t cie,
                                    CIE_Info *cieInfo) {
  cieInfo->pointerEncoding = 0;
  cieInfo->lsdaEncoding = DW_EH_PE_omit;
  cieInfo->personalityEncoding = 0;
  cieInfo->personalityOffsetInCIE = 0;
  cieInfo->personality = 0;
  cieInfo->codeAlignFactor = 0;
  cieInfo->dataAlignFactor = 0;
  cieInfo->isSignalFrame = false;
  cieInfo->fdesHaveAugmentationData = false;
#if defined(_LIBUNWIND_TARGET_AARCH64)
  cieInfo->addressesSignedWithBKey = false;
#endif
  cieInfo->cieStart = cie;
  pint_t p = cie;
  pint_t cieLength = (pint_t)addressSpace.get32(p);
  p += 4;
  pint_t cieContentEnd = p + cieLength;
  if (cieLength == 0xffffffff) {
    // 0xffffffff means length is really next 8 bytes
    cieLength = (pint_t)addressSpace.get64(p);
    p += 8;
    cieContentEnd = p + cieLength;
  }
  if (cieLength == 0)
    return NULL;
  // CIE ID is always 0
  if (addressSpace.get32(p) != 0)
    return "CIE ID is not zero";
  p += 4;
  // Version is always 1 or 3
  uint8_t version = addressSpace.get8(p);
  if ((version != 1) && (version != 3))
    return "CIE version is not 1 or 3";
  ++p;
  // save start of augmentation string and find end
  pint_t strStart = p;
  while (addressSpace.get8(p) != 0)
    ++p;
  ++p;
  // parse code aligment factor
  cieInfo->codeAlignFactor = (uint32_t)addressSpace.getULEB128(p, cieContentEnd);
  // parse data alignment factor
  cieInfo->dataAlignFactor = (int)addressSpace.getSLEB128(p, cieContentEnd);
  // parse return address register
  uint64_t raReg = addressSpace.getULEB128(p, cieContentEnd);
  assert(raReg < 255 && "return address register too large");
  cieInfo->returnAddressRegister = (uint8_t)raReg;
  // parse augmentation data based on augmentation string
  const char *result = NULL;
  if (addressSpace.get8(strStart) == 'z') {
    // parse augmentation data length
    addressSpace.getULEB128(p, cieContentEnd);
    for (pint_t s = strStart; addressSpace.get8(s) != '\0'; ++s) {
      switch (addressSpace.get8(s)) {
      case 'z':
        cieInfo->fdesHaveAugmentationData = true;
        break;
      case 'P':
        cieInfo->personalityEncoding = addressSpace.get8(p);
        ++p;
        cieInfo->personalityOffsetInCIE = (uint8_t)(p - cie);
        cieInfo->personality = addressSpace
            .getEncodedP(p, cieContentEnd, cieInfo->personalityEncoding);
        break;
      case 'L':
        cieInfo->lsdaEncoding = addressSpace.get8(p);
        ++p;
        break;
      case 'R':
        cieInfo->pointerEncoding = addressSpace.get8(p);
        ++p;
        break;
      case 'S':
        cieInfo->isSignalFrame = true;
        break;
#if defined(_LIBUNWIND_TARGET_AARCH64)
      case 'B':
        cieInfo->addressesSignedWithBKey = true;
        break;
#endif
      default:
        // ignore unknown letters
        break;
      }
    }
  }
  cieInfo->cieLength = cieContentEnd - cieInfo->cieStart;
  cieInfo->cieInstructions = p;
  return result;
}


/// "run" the DWARF instructions and create the abstact PrologInfo for an FDE
template <typename A>
bool CFI_Parser<A>::parseFDEInstructions(A &addressSpace,
                                         const FDE_Info &fdeInfo,
                                         const CIE_Info &cieInfo, pint_t upToPC,
                                         int arch, PrologInfo *results) {
  // clear results
  memset(results, '\0', sizeof(PrologInfo));
  results->cfaRegister = (uint32_t)(-1);

  PrologInfoStackEntry *rememberStack = NULL;

  StackGuard stack;

  // parse CIE then FDE instructions
  return parseInstructions(addressSpace, cieInfo.cieInstructions,
                           cieInfo.cieStart + cieInfo.cieLength, cieInfo,
                           (pint_t)(-1), rememberStack, arch, results, stack) &&
         parseInstructions(addressSpace, fdeInfo.fdeInstructions,
                           fdeInfo.fdeStart + fdeInfo.fdeLength, cieInfo,
                           upToPC - fdeInfo.pcStart, rememberStack, arch,
                           results, stack);
}

/// "run" the DWARF instructions
template <typename A>
bool CFI_Parser<A>::parseInstructions(A &addressSpace, pint_t instructions,
                                      pint_t instructionsEnd,
                                      const CIE_Info &cieInfo, pint_t pcoffset,
                                      PrologInfoStackEntry *&rememberStack,
                                      int arch, PrologInfo *results,
                                      StackGuard &stack) {
  pint_t p = instructions;
  pint_t codeOffset = 0;
  PrologInfo initialState = *results;

  _LIBUNWIND_TRACE_DWARF("parseInstructions(instructions=0x%0" PRIx64 ")\n",
                         static_cast<uint64_t>(instructionsEnd));

  // see DWARF Spec, section 6.4.2 for details on unwind opcodes
  while ((p < instructionsEnd) && (codeOffset < pcoffset)) {
    uint64_t reg;
    uint64_t reg2;
    int64_t offset;
    uint64_t length;
    uint8_t opcode = addressSpace.get8(p);
    uint8_t operand;
    PrologInfoStackEntry *entry;
    ++p;
    switch (opcode) {
    case DW_CFA_nop:
      _LIBUNWIND_TRACE_DWARF("DW_CFA_nop\n");
      break;
    case DW_CFA_set_loc:
      codeOffset =
          addressSpace.getEncodedP(p, instructionsEnd, cieInfo.pointerEncoding);
      _LIBUNWIND_TRACE_DWARF("DW_CFA_set_loc\n");
      break;
    case DW_CFA_advance_loc1:
      codeOffset += (addressSpace.get8(p) * cieInfo.codeAlignFactor);
      p += 1;
      _LIBUNWIND_TRACE_DWARF("DW_CFA_advance_loc1: new offset=%" PRIu64 "\n",
                             static_cast<uint64_t>(codeOffset));
      break;
    case DW_CFA_advance_loc2:
      codeOffset += (addressSpace.get16(p) * cieInfo.codeAlignFactor);
      p += 2;
      _LIBUNWIND_TRACE_DWARF("DW_CFA_advance_loc2: new offset=%" PRIu64 "\n",
                             static_cast<uint64_t>(codeOffset));
      break;
    case DW_CFA_advance_loc4:
      codeOffset += (addressSpace.get32(p) * cieInfo.codeAlignFactor);
      p += 4;
      _LIBUNWIND_TRACE_DWARF("DW_CFA_advance_loc4: new offset=%" PRIu64 "\n",
                             static_cast<uint64_t>(codeOffset));
      break;
    case DW_CFA_offset_extended:
      reg = addressSpace.getULEB128(p, instructionsEnd);
      offset = (int64_t)addressSpace.getULEB128(p, instructionsEnd)
                                                  * cieInfo.dataAlignFactor;
      if (reg > kMaxRegisterNumber) {
        _LIBUNWIND_LOG0(
                "malformed DW_CFA_offset_extended DWARF unwind, reg too big");
        return false;
      }
      results->savedRegisters[reg].location = kRegisterInCFA;
      results->savedRegisters[reg].value = offset;
      _LIBUNWIND_TRACE_DWARF("DW_CFA_offset_extended(reg=%" PRIu64 ", "
                             "offset=%" PRId64 ")\n",
                             reg, offset);
      break;
    case DW_CFA_restore_extended:
      reg = addressSpace.getULEB128(p, instructionsEnd);
      if (reg > kMaxRegisterNumber) {
        _LIBUNWIND_LOG0(
            "malformed DW_CFA_restore_extended DWARF unwind, reg too big");
        return false;
      }
      results->savedRegisters[reg] = initialState.savedRegisters[reg];
      _LIBUNWIND_TRACE_DWARF("DW_CFA_restore_extended(reg=%" PRIu64 ")\n", reg);
      break;
    case DW_CFA_undefined:
      reg = addressSpace.getULEB128(p, instructionsEnd);
      if (reg > kMaxRegisterNumber) {
        _LIBUNWIND_LOG0(
                "malformed DW_CFA_undefined DWARF unwind, reg too big");
        return false;
      }
      results->savedRegisters[reg].location = kRegisterUnused;
      _LIBUNWIND_TRACE_DWARF("DW_CFA_undefined(reg=%" PRIu64 ")\n", reg);
      break;
    case DW_CFA_same_value:
      reg = addressSpace.getULEB128(p, instructionsEnd);
      if (reg > kMaxRegisterNumber) {
        _LIBUNWIND_LOG0(
                "malformed DW_CFA_same_value DWARF unwind, reg too big");
        return false;
      }
      // <rdar://problem/8456377> DW_CFA_same_value unsupported
      // "same value" means register was stored in frame, but its current
      // value has not changed, so no need to restore from frame.
      // We model this as if the register was never saved.
      results->savedRegisters[reg].location = kRegisterUnused;
      // set flag to disable conversion to compact unwind
      results->sameValueUsed = true;
      _LIBUNWIND_TRACE_DWARF("DW_CFA_same_value(reg=%" PRIu64 ")\n", reg);
      break;
    case DW_CFA_register:
      reg = addressSpace.getULEB128(p, instructionsEnd);
      reg2 = addressSpace.getULEB128(p, instructionsEnd);
      if (reg > kMaxRegisterNumber) {
        _LIBUNWIND_LOG0(
                "malformed DW_CFA_register DWARF unwind, reg too big");
        return false;
      }
      if (reg2 > kMaxRegisterNumber) {
        _LIBUNWIND_LOG0(
                "malformed DW_CFA_register DWARF unwind, reg2 too big");
        return false;
      }
      results->savedRegisters[reg].location = kRegisterInRegister;
      results->savedRegisters[reg].value = (int64_t)reg2;
      // set flag to disable conversion to compact unwind
      results->registersInOtherRegisters = true;
      _LIBUNWIND_TRACE_DWARF(
          "DW_CFA_register(reg=%" PRIu64 ", reg2=%" PRIu64 ")\n", reg, reg2);
      break;
    case DW_CFA_remember_state:
      // avoid operator new, because that would be an upward dependency
      entry = (PrologInfoStackEntry *)stack.push();
      if (entry != NULL) {
        entry->next = rememberStack;
        entry->info = *results;
        rememberStack = entry;
      } else {
        return false;
      }
      _LIBUNWIND_TRACE_DWARF("DW_CFA_remember_state\n");
      break;
    case DW_CFA_restore_state:
      if (rememberStack != NULL) {
        PrologInfoStackEntry *top = rememberStack;
        *results = top->info;
        rememberStack = top->next;
        stack.pop();
      } else {
        return false;
      }
      _LIBUNWIND_TRACE_DWARF("DW_CFA_restore_state\n");
      break;
    case DW_CFA_def_cfa:
      reg = addressSpace.getULEB128(p, instructionsEnd);
      offset = (int64_t)addressSpace.getULEB128(p, instructionsEnd);
      if (reg > kMaxRegisterNumber) {
        _LIBUNWIND_LOG0("malformed DW_CFA_def_cfa DWARF unwind, reg too big");
        return false;
      }
      results->cfaRegister = (uint32_t)reg;
      results->cfaRegisterOffset = (int32_t)offset;
      _LIBUNWIND_TRACE_DWARF(
          "DW_CFA_def_cfa(reg=%" PRIu64 ", offset=%" PRIu64 ")\n", reg, offset);
      break;
    case DW_CFA_def_cfa_register:
      reg = addressSpace.getULEB128(p, instructionsEnd);
      if (reg > kMaxRegisterNumber) {
        _LIBUNWIND_LOG0(
            "malformed DW_CFA_def_cfa_register DWARF unwind, reg too big");
        return false;
      }
      results->cfaRegister = (uint32_t)reg;
      _LIBUNWIND_TRACE_DWARF("DW_CFA_def_cfa_register(%" PRIu64 ")\n", reg);
      break;
    case DW_CFA_def_cfa_offset:
      results->cfaRegisterOffset = (int32_t)
                                  addressSpace.getULEB128(p, instructionsEnd);
      results->codeOffsetAtStackDecrement = (uint32_t)codeOffset;
      _LIBUNWIND_TRACE_DWARF("DW_CFA_def_cfa_offset(%d)\n",
                             results->cfaRegisterOffset);
      break;
    case DW_CFA_def_cfa_expression:
      results->cfaRegister = (uint32_t)(-1);
      results->cfaExpression = (int64_t)p;
      length = addressSpace.getULEB128(p, instructionsEnd);
      assert(length < static_cast<pint_t>(~0) && "pointer overflow");
      p += static_cast<pint_t>(length);
      _LIBUNWIND_TRACE_DWARF("DW_CFA_def_cfa_expression(expression=0x%" PRIx64
                             ", length=%" PRIu64 ")\n",
                             results->cfaExpression, length);
      break;
    case DW_CFA_expression:
      reg = addressSpace.getULEB128(p, instructionsEnd);
      if (reg > kMaxRegisterNumber) {
        _LIBUNWIND_LOG0(
                "malformed DW_CFA_expression DWARF unwind, reg too big");
        return false;
      }
      results->savedRegisters[reg].location = kRegisterAtExpression;
      results->savedRegisters[reg].value = (int64_t)p;
      length = addressSpace.getULEB128(p, instructionsEnd);
      assert(length < static_cast<pint_t>(~0) && "pointer overflow");
      p += static_cast<pint_t>(length);
      _LIBUNWIND_TRACE_DWARF("DW_CFA_expression(reg=%" PRIu64 ", "
                             "expression=0x%" PRIx64 ", "
                             "length=%" PRIu64 ")\n",
                             reg, results->savedRegisters[reg].value, length);
      break;
    case DW_CFA_offset_extended_sf:
      reg = addressSpace.getULEB128(p, instructionsEnd);
      if (reg > kMaxRegisterNumber) {
        _LIBUNWIND_LOG0(
            "malformed DW_CFA_offset_extended_sf DWARF unwind, reg too big");
        return false;
      }
      offset =
          addressSpace.getSLEB128(p, instructionsEnd) * cieInfo.dataAlignFactor;
      results->savedRegisters[reg].location = kRegisterInCFA;
      results->savedRegisters[reg].value = offset;
      _LIBUNWIND_TRACE_DWARF("DW_CFA_offset_extended_sf(reg=%" PRIu64 ", "
                             "offset=%" PRId64 ")\n",
                             reg, offset);
      break;
    case DW_CFA_def_cfa_sf:
      reg = addressSpace.getULEB128(p, instructionsEnd);
      offset =
          addressSpace.getSLEB128(p, instructionsEnd) * cieInfo.dataAlignFactor;
      if (reg > kMaxRegisterNumber) {
        _LIBUNWIND_LOG0(
                "malformed DW_CFA_def_cfa_sf DWARF unwind, reg too big");
        return false;
      }
      results->cfaRegister = (uint32_t)reg;
      results->cfaRegisterOffset = (int32_t)offset;
      _LIBUNWIND_TRACE_DWARF("DW_CFA_def_cfa_sf(reg=%" PRIu64 ", "
                             "offset=%" PRId64 ")\n",
                             reg, offset);
      break;
    case DW_CFA_def_cfa_offset_sf:
      results->cfaRegisterOffset = (int32_t)
        (addressSpace.getSLEB128(p, instructionsEnd) * cieInfo.dataAlignFactor);
      results->codeOffsetAtStackDecrement = (uint32_t)codeOffset;
      _LIBUNWIND_TRACE_DWARF("DW_CFA_def_cfa_offset_sf(%d)\n",
                             results->cfaRegisterOffset);
      break;
    case DW_CFA_val_offset:
      reg = addressSpace.getULEB128(p, instructionsEnd);
      if (reg > kMaxRegisterNumber) {
        _LIBUNWIND_LOG(
                "malformed DW_CFA_val_offset DWARF unwind, reg (%" PRIu64
                ") out of range\n",
                reg);
        return false;
      }
      offset = (int64_t)addressSpace.getULEB128(p, instructionsEnd)
                                                    * cieInfo.dataAlignFactor;
      results->savedRegisters[reg].location = kRegisterOffsetFromCFA;
      results->savedRegisters[reg].value = offset;
      _LIBUNWIND_TRACE_DWARF("DW_CFA_val_offset(reg=%" PRIu64 ", "
                             "offset=%" PRId64 "\n",
                             reg, offset);
      break;
    case DW_CFA_val_offset_sf:
      reg = addressSpace.getULEB128(p, instructionsEnd);
      if (reg > kMaxRegisterNumber) {
        _LIBUNWIND_LOG0(
                "malformed DW_CFA_val_offset_sf DWARF unwind, reg too big");
        return false;
      }
      offset =
          addressSpace.getSLEB128(p, instructionsEnd) * cieInfo.dataAlignFactor;
      results->savedRegisters[reg].location = kRegisterOffsetFromCFA;
      results->savedRegisters[reg].value = offset;
      _LIBUNWIND_TRACE_DWARF("DW_CFA_val_offset_sf(reg=%" PRIu64 ", "
                             "offset=%" PRId64 "\n",
                             reg, offset);
      break;
    case DW_CFA_val_expression:
      reg = addressSpace.getULEB128(p, instructionsEnd);
      if (reg > kMaxRegisterNumber) {
        _LIBUNWIND_LOG0(
                "malformed DW_CFA_val_expression DWARF unwind, reg too big");
        return false;
      }
      results->savedRegisters[reg].location = kRegisterIsExpression;
      results->savedRegisters[reg].value = (int64_t)p;
      length = addressSpace.getULEB128(p, instructionsEnd);
      assert(length < static_cast<pint_t>(~0) && "pointer overflow");
      p += static_cast<pint_t>(length);
      _LIBUNWIND_TRACE_DWARF("DW_CFA_val_expression(reg=%" PRIu64 ", "
                             "expression=0x%" PRIx64 ", length=%" PRIu64 ")\n",
                             reg, results->savedRegisters[reg].value, length);
      break;
    case DW_CFA_GNU_args_size:
      length = addressSpace.getULEB128(p, instructionsEnd);
      results->spExtraArgSize = (uint32_t)length;
      _LIBUNWIND_TRACE_DWARF("DW_CFA_GNU_args_size(%" PRIu64 ")\n", length);
      break;
    case DW_CFA_GNU_negative_offset_extended:
      reg = addressSpace.getULEB128(p, instructionsEnd);
      if (reg > kMaxRegisterNumber) {
        _LIBUNWIND_LOG0("malformed DW_CFA_GNU_negative_offset_extended DWARF "
                        "unwind, reg too big");
        return false;
      }
      offset = (int64_t)addressSpace.getULEB128(p, instructionsEnd)
                                                    * cieInfo.dataAlignFactor;
      results->savedRegisters[reg].location = kRegisterInCFA;
      results->savedRegisters[reg].value = -offset;
      _LIBUNWIND_TRACE_DWARF(
          "DW_CFA_GNU_negative_offset_extended(%" PRId64 ")\n", offset);
      break;

#if defined(_LIBUNWIND_TARGET_AARCH64) || defined(_LIBUNWIND_TARGET_SPARC)
    // The same constant is used to represent different instructions on
    // AArch64 (negate_ra_state) and SPARC (window_save).
    static_assert(DW_CFA_AARCH64_negate_ra_state == DW_CFA_GNU_window_save,
                  "uses the same constant");
    case DW_CFA_AARCH64_negate_ra_state:
      switch (arch) {
#if defined(_LIBUNWIND_TARGET_AARCH64)
      case REGISTERS_ARM64:
        results->savedRegisters[UNW_ARM64_RA_SIGN_STATE].value ^= 0x1;
        _LIBUNWIND_TRACE_DWARF("DW_CFA_AARCH64_negate_ra_state\n");
        break;
#endif
#if defined(_LIBUNWIND_TARGET_SPARC)
      // case DW_CFA_GNU_window_save:
      case REGISTERS_SPARC:
        _LIBUNWIND_TRACE_DWARF("DW_CFA_GNU_window_save()\n");
        for (reg = UNW_SPARC_O0; reg <= UNW_SPARC_O7; reg++) {
          results->savedRegisters[reg].location = kRegisterInRegister;
          results->savedRegisters[reg].value =
              ((int64_t)reg - UNW_SPARC_O0) + UNW_SPARC_I0;
        }

        for (reg = UNW_SPARC_L0; reg <= UNW_SPARC_I7; reg++) {
          results->savedRegisters[reg].location = kRegisterInCFA;
          results->savedRegisters[reg].value =
              ((int64_t)reg - UNW_SPARC_L0) * 4;
        }
        break;
#endif
      }
      break;
#else
      (void)arch;
#endif

    default:
      operand = opcode & 0x3F;
      switch (opcode & 0xC0) {
      case DW_CFA_offset:
        reg = operand;
        if (reg > kMaxRegisterNumber) {
          _LIBUNWIND_LOG("malformed DW_CFA_offset DWARF unwind, reg (%" PRIu64
                         ") out of range",
                  reg);
          return false;
        }
        offset = (int64_t)addressSpace.getULEB128(p, instructionsEnd)
                                                    * cieInfo.dataAlignFactor;
        results->savedRegisters[reg].location = kRegisterInCFA;
        results->savedRegisters[reg].value = offset;
        _LIBUNWIND_TRACE_DWARF("DW_CFA_offset(reg=%d, offset=%" PRId64 ")\n",
                               operand, offset);
        break;
      case DW_CFA_advance_loc:
        codeOffset += operand * cieInfo.codeAlignFactor;
        _LIBUNWIND_TRACE_DWARF("DW_CFA_advance_loc: new offset=%" PRIu64 "\n",
                               static_cast<uint64_t>(codeOffset));
        break;
      case DW_CFA_restore:
        reg = operand;
        if (reg > kMaxRegisterNumber) {
          _LIBUNWIND_LOG("malformed DW_CFA_restore DWARF unwind, reg (%" PRIu64
                         ") out of range",
                  reg);
          return false;
        }
        results->savedRegisters[reg] = initialState.savedRegisters[reg];
        _LIBUNWIND_TRACE_DWARF("DW_CFA_restore(reg=%" PRIu64 ")\n",
                               static_cast<uint64_t>(operand));
        break;
      default:
        _LIBUNWIND_TRACE_DWARF("unknown CFA opcode 0x%02X\n", opcode);
        return false;
      }
    }
  }

  return true;
}

} // namespace libunwind

#endif // __DWARF_PARSER_HPP__
