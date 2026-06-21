// Copyright (c) Rux contributors.
// SPDX-License-Identifier: MIT

#include "Rux/Rcu.h"
#include "Rux/Version.h"
#include "RcuShared.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <format>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Rux {
    namespace {

        // RISC-V Register numbers
        constexpr uint8_t RV_X0  = 0;  // zero
        constexpr uint8_t RV_RA  = 1;  // return address
        constexpr uint8_t RV_SP  = 2;  // stack pointer
        constexpr uint8_t RV_GP  = 3;  // global pointer
        constexpr uint8_t RV_TP  = 4;  // thread pointer
        constexpr uint8_t RV_T0  = 5;  // temp / LoadA register
        constexpr uint8_t RV_T1  = 6;  // temp / LoadB register
        constexpr uint8_t RV_T2  = 7;  // temp / pointer deref
        constexpr uint8_t RV_S0  = 8;  // frame pointer
        constexpr uint8_t RV_S1  = 9;  // saved
        constexpr uint8_t RV_A0  = 10; // arg / return
        constexpr uint8_t RV_A1  = 11;
        constexpr uint8_t RV_A2  = 12;
        constexpr uint8_t RV_A3  = 13;
        constexpr uint8_t RV_A4  = 14;
        constexpr uint8_t RV_A5  = 15;
        constexpr uint8_t RV_A6  = 16;
        constexpr uint8_t RV_A7  = 17;
        constexpr uint8_t RV_S2  = 18;
        constexpr uint8_t RV_S3  = 19;
        constexpr uint8_t RV_S4  = 20;
        constexpr uint8_t RV_S5  = 21;
        constexpr uint8_t RV_S6  = 22;
        constexpr uint8_t RV_S7  = 23;
        constexpr uint8_t RV_S8  = 24;
        constexpr uint8_t RV_S9  = 25;
        constexpr uint8_t RV_S10 = 26;
        constexpr uint8_t RV_S11 = 27;
        constexpr uint8_t RV_T3  = 28;

        // RISC-V FP registers
        constexpr uint8_t RV_FT0 = 0;
        constexpr uint8_t RV_FA0 = 10; // float arg / return

        // RISC-V opcodes
        constexpr uint8_t OP_IMM  = 0x13;
        constexpr uint8_t OP      = 0x33;
        constexpr uint8_t LUI     = 0x37;
        constexpr uint8_t AUIPC   = 0x17;
        constexpr uint8_t JAL     = 0x6F;
        constexpr uint8_t JALR    = 0x67;
        constexpr uint8_t BRANCH  = 0x63;
        constexpr uint8_t LOAD    = 0x03;
        constexpr uint8_t STORE   = 0x23;
        constexpr uint8_t FLOAD   = 0x07;
        constexpr uint8_t FSTORE  = 0x27;
        constexpr uint8_t OP_FP   = 0x53;
        constexpr uint8_t OP_32   = 0x3B;
        constexpr uint8_t OP_IMM_32 = 0x1B;
        constexpr uint8_t MUL_OP  = 0x33; // same as OP with funct7=1
        constexpr uint8_t SYSTEM  = 0x73;

        // funct3 for loads/stores
        constexpr uint8_t F3_LB  = 0;
        constexpr uint8_t F3_LH  = 1;
        constexpr uint8_t F3_LW  = 2;
        constexpr uint8_t F3_LD  = 3;
        constexpr uint8_t F3_LBU = 4;
        constexpr uint8_t F3_LHU = 5;
        constexpr uint8_t F3_LWU = 6;
        constexpr uint8_t F3_SB  = 0;
        constexpr uint8_t F3_SH  = 1;
        constexpr uint8_t F3_SW  = 2;
        constexpr uint8_t F3_SD  = 3;
        constexpr uint8_t F3_FLW = 2;
        constexpr uint8_t F3_FLD = 3;
        constexpr uint8_t F3_FSW = 2;
        constexpr uint8_t F3_FSD = 3;

        // funct3 for ALU immediate
        constexpr uint8_t F3_ADDI  = 0;
        constexpr uint8_t F3_SLLI  = 1;
        constexpr uint8_t F3_SLTI  = 2;
        constexpr uint8_t F3_SLTIU = 3;
        constexpr uint8_t F3_XORI  = 4;
        constexpr uint8_t F3_SRLI  = 5;
        constexpr uint8_t F3_ORI   = 6;
        constexpr uint8_t F3_ANDI  = 7;

        // funct3 for ALU reg
        constexpr uint8_t F3_ADD  = 0;
        constexpr uint8_t F3_SLL  = 1;
        constexpr uint8_t F3_SLT  = 2;
        constexpr uint8_t F3_SLTU = 3;
        constexpr uint8_t F3_XOR  = 4;
        constexpr uint8_t F3_SRL  = 5;
        constexpr uint8_t F3_OR   = 6;
        constexpr uint8_t F3_AND  = 7;
        constexpr uint8_t F3_MUL  = 0;
        constexpr uint8_t F3_DIV  = 4;
        constexpr uint8_t F3_REM  = 6;

        // funct3 for branches
        constexpr uint8_t F3_BEQ  = 0;
        constexpr uint8_t F3_BNE  = 1;
        constexpr uint8_t F3_BLT  = 4;
        constexpr uint8_t F3_BGE  = 5;
        constexpr uint8_t F3_BLTU = 6;
        constexpr uint8_t F3_BGEU = 7;

        // funct7 values
        constexpr uint8_t F7_ADD  = 0x00;
        constexpr uint8_t F7_SUB  = 0x20;
        constexpr uint8_t F7_MUL  = 0x01;
        constexpr uint8_t F7_DIV  = 0x01;
        constexpr uint8_t F7_REM  = 0x01;

        // Float funct7 for OP_FP (funct5 | (fmt<<2))
        constexpr uint8_t F7_FADD_S = 0x00;
        constexpr uint8_t F7_FSUB_S = 0x04;
        constexpr uint8_t F7_FMUL_S = 0x08;
        constexpr uint8_t F7_FDIV_S = 0x0C;
        constexpr uint8_t F7_FSQRT_S = 0x2C;
        constexpr uint8_t F7_FADD_D = 0x01;
        constexpr uint8_t F7_FSUB_D = 0x05;
        constexpr uint8_t F7_FMUL_D = 0x09;
        constexpr uint8_t F7_FDIV_D = 0x0D;
        constexpr uint8_t F7_FSQRT_D = 0x2D;
        constexpr uint8_t F7_FSGNJ_S = 0x10;
        constexpr uint8_t F7_FSGNJ_D = 0x11;
        constexpr uint8_t F7_FCVT_S_D = 0x20;
        constexpr uint8_t F7_FCVT_D_S = 0x21;
        constexpr uint8_t F7_FCVT_W_S = 0x60;
        constexpr uint8_t F7_FCVT_W_D = 0x61;
        constexpr uint8_t F7_FCVT_L_S = 0x68;
        constexpr uint8_t F7_FCVT_L_D = 0x69;
        constexpr uint8_t F7_FMV_X_W = 0x70;
        constexpr uint8_t F7_FMV_X_D = 0x71;
        constexpr uint8_t F7_FEQ_S = 0x50;
        constexpr uint8_t F7_FEQ_D = 0x51;
        constexpr uint8_t F7_FLT_S = 0x54;
        constexpr uint8_t F7_FLT_D = 0x55;
        constexpr uint8_t F7_FLE_S = 0x58;
        constexpr uint8_t F7_FLE_D = 0x59;
        constexpr uint8_t F7_FCLASS_S = 0x74;
        constexpr uint8_t F7_FCLASS_D = 0x75;

        // RISC-V instruction format helpers
        static uint32_t RvR(uint8_t funct7, uint8_t rs2, uint8_t rs1,
                           uint8_t funct3, uint8_t rd, uint8_t opcode) {
            return (static_cast<uint32_t>(funct7) << 25) |
                   (static_cast<uint32_t>(rs2 & 0x1F) << 20) |
                   (static_cast<uint32_t>(rs1 & 0x1F) << 15) |
                   (static_cast<uint32_t>(funct3 & 0x7) << 12) |
                   (static_cast<uint32_t>(rd & 0x1F) << 7) |
                   (opcode & 0x7F);
        }

        static uint32_t RvI(uint32_t imm12, uint8_t rs1,
                           uint8_t funct3, uint8_t rd, uint8_t opcode) {
            return ((imm12 & 0xFFF) << 20) |
                   (static_cast<uint32_t>(rs1 & 0x1F) << 15) |
                   (static_cast<uint32_t>(funct3 & 0x7) << 12) |
                   (static_cast<uint32_t>(rd & 0x1F) << 7) |
                   (opcode & 0x7F);
        }

        static uint32_t RvS(uint32_t imm12, uint8_t rs2,
                           uint8_t rs1, uint8_t funct3, uint8_t opcode) {
            return (((imm12 >> 5) & 0x7F) << 25) |
                   (static_cast<uint32_t>(rs2 & 0x1F) << 20) |
                   (static_cast<uint32_t>(rs1 & 0x1F) << 15) |
                   (static_cast<uint32_t>(funct3 & 0x7) << 12) |
                   ((imm12 & 0x1F) << 7) |
                   (opcode & 0x7F);
        }

        static uint32_t RvB(uint32_t imm13, uint8_t rs2,
                           uint8_t rs1, uint8_t funct3, uint8_t opcode) {
            imm13 &= 0x1FFF;
            const uint32_t b12 = (imm13 >> 12) & 1;
            const uint32_t b10_5 = (imm13 >> 5) & 0x3F;
            const uint32_t b4_1 = (imm13 >> 1) & 0x0F;
            const uint32_t b11 = (imm13 >> 11) & 1;
            return (b12 << 31) |
                   (b10_5 << 25) |
                   (static_cast<uint32_t>(rs2 & 0x1F) << 20) |
                   (static_cast<uint32_t>(rs1 & 0x1F) << 15) |
                   (static_cast<uint32_t>(funct3 & 0x7) << 12) |
                   (b4_1 << 8) |
                   (b11 << 7) |
                   (opcode & 0x7F);
        }

        static uint32_t RvU(uint32_t imm20, uint8_t rd, uint8_t opcode) {
            return ((imm20 & 0xFFFFF) << 12) |
                   (static_cast<uint32_t>(rd & 0x1F) << 7) |
                   (opcode & 0x7F);
        }

        static uint32_t RvJ(uint32_t imm21, uint8_t rd, uint8_t opcode) {
            imm21 &= 0x1FFFFF;
            const uint32_t j20 = (imm21 >> 20) & 1;
            const uint32_t j10_1 = (imm21 >> 1) & 0x3FF;
            const uint32_t j11 = (imm21 >> 11) & 1;
            const uint32_t j19_12 = (imm21 >> 12) & 0xFF;
            return (j20 << 31) |
                   (j10_1 << 21) |
                   (j11 << 20) |
                   (j19_12 << 12) |
                   (static_cast<uint32_t>(rd & 0x1F) << 7) |
                   (opcode & 0x7F);
        }

        class RiscV64Enc {
        public:
            explicit RiscV64Enc(std::vector<uint8_t>& buf) : out_(buf) {}

            [[nodiscard]] uint32_t Size() const {
                return static_cast<uint32_t>(out_.size());
            }

            void Instr(uint32_t v) const {
                out_.push_back(v & 0xFF);
                out_.push_back((v >> 8) & 0xFF);
                out_.push_back((v >> 16) & 0xFF);
                out_.push_back((v >> 24) & 0xFF);
            }

            void Patch32(uint32_t off, uint32_t v) const {
                out_[off] = v & 0xFF;
                out_[off + 1] = (v >> 8) & 0xFF;
                out_[off + 2] = (v >> 16) & 0xFF;
                out_[off + 3] = (v >> 24) & 0xFF;
            }

            void PatchAt(uint32_t off, uint32_t v) const {
                Patch32(off, v);
            }

            void Nop() const { Instr(RvI(0, 0, 0, 0, OP_IMM)); }

            // Integer ALU (R-type)
            void Add(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(F7_ADD, rs2, rs1, F3_ADD, rd, OP));
            }
            void Sub(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(F7_SUB, rs2, rs1, F3_ADD, rd, OP));
            }
            void Sll(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(0, rs2, rs1, F3_SLL, rd, OP));
            }
            void Srl(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(0, rs2, rs1, F3_SRL, rd, OP));
            }
            void Sra(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(F7_SUB, rs2, rs1, F3_SRL, rd, OP));
            }
            void And_(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(0, rs2, rs1, F3_AND, rd, OP));
            }
            void Or_(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(0, rs2, rs1, F3_OR, rd, OP));
            }
            void Xor_(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(0, rs2, rs1, F3_XOR, rd, OP));
            }
            void Slt(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(0, rs2, rs1, F3_SLT, rd, OP));
            }
            void Sltu(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(0, rs2, rs1, F3_SLTU, rd, OP));
            }
            void Mul(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(F7_MUL, rs2, rs1, F3_MUL, rd, OP));
            }
            void Div(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(F7_DIV, rs2, rs1, F3_DIV, rd, OP));
            }
            void Divu(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(F7_DIV, rs2, rs1, F3_DIV | 1, rd, OP));
            }
            void Rem(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(F7_REM, rs2, rs1, F3_REM, rd, OP));
            }
            void Remu(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(F7_REM, rs2, rs1, F3_REM | 1, rd, OP));
            }

            // Integer ALU immediate
            void Addi(uint8_t rd, uint8_t rs1, int32_t imm12) const {
                Instr(RvI(static_cast<uint32_t>(imm12), rs1, F3_ADDI, rd, OP_IMM));
            }
            void Addiw(uint8_t rd, uint8_t rs1, int32_t imm12) const {
                Instr(RvI(static_cast<uint32_t>(imm12), rs1, F3_ADDI, rd, OP_IMM_32));
            }
            void Slli(uint8_t rd, uint8_t rs1, uint32_t shamt) const {
                Instr(RvI(shamt & 0x3F, rs1, F3_SLLI, rd, OP_IMM));
            }
            void Srli(uint8_t rd, uint8_t rs1, uint32_t shamt) const {
                Instr(RvI(shamt & 0x3F, rs1, F3_SRLI, rd, OP_IMM));
            }
            void Srai(uint8_t rd, uint8_t rs1, uint32_t shamt) const {
                Instr(RvI((shamt & 0x3F) | 0x400, rs1, F3_SRLI, rd, OP_IMM));
            }
            void Andi(uint8_t rd, uint8_t rs1, int32_t imm12) const {
                Instr(RvI(static_cast<uint32_t>(imm12), rs1, F3_ANDI, rd, OP_IMM));
            }
            void Ori(uint8_t rd, uint8_t rs1, int32_t imm12) const {
                Instr(RvI(static_cast<uint32_t>(imm12), rs1, F3_ORI, rd, OP_IMM));
            }
            void Xori(uint8_t rd, uint8_t rs1, int32_t imm12) const {
                Instr(RvI(static_cast<uint32_t>(imm12), rs1, F3_XORI, rd, OP_IMM));
            }
            void Slti(uint8_t rd, uint8_t rs1, int32_t imm12) const {
                Instr(RvI(static_cast<uint32_t>(imm12), rs1, F3_SLTI, rd, OP_IMM));
            }
            void Sltiu(uint8_t rd, uint8_t rs1, int32_t imm12) const {
                Instr(RvI(static_cast<uint32_t>(imm12), rs1, F3_SLTIU, rd, OP_IMM));
            }

            // LUI / AUIPC
            void Lui(uint8_t rd, uint32_t imm20) const {
                Instr(RvU(imm20, rd, LUI));
            }
            void Auipc(uint8_t rd, uint32_t imm20) const {
                Instr(RvU(imm20, rd, AUIPC));
            }

            // Loads (I-type)
            void Lb(uint8_t rd, uint8_t rs1, int32_t imm12) const {
                Instr(RvI(static_cast<uint32_t>(imm12), rs1, F3_LB, rd, LOAD));
            }
            void Lh(uint8_t rd, uint8_t rs1, int32_t imm12) const {
                Instr(RvI(static_cast<uint32_t>(imm12), rs1, F3_LH, rd, LOAD));
            }
            void Lw(uint8_t rd, uint8_t rs1, int32_t imm12) const {
                Instr(RvI(static_cast<uint32_t>(imm12), rs1, F3_LW, rd, LOAD));
            }
            void Ld(uint8_t rd, uint8_t rs1, int32_t imm12) const {
                Instr(RvI(static_cast<uint32_t>(imm12), rs1, F3_LD, rd, LOAD));
            }
            void Lbu(uint8_t rd, uint8_t rs1, int32_t imm12) const {
                Instr(RvI(static_cast<uint32_t>(imm12), rs1, F3_LBU, rd, LOAD));
            }
            void Lhu(uint8_t rd, uint8_t rs1, int32_t imm12) const {
                Instr(RvI(static_cast<uint32_t>(imm12), rs1, F3_LHU, rd, LOAD));
            }
            void Lwu(uint8_t rd, uint8_t rs1, int32_t imm12) const {
                Instr(RvI(static_cast<uint32_t>(imm12), rs1, F3_LWU, rd, LOAD));
            }

            // Stores (S-type)
            void Sb(uint8_t rs2, uint8_t rs1, int32_t imm12) const {
                Instr(RvS(static_cast<uint32_t>(imm12), rs2, rs1, F3_SB, STORE));
            }
            void Sh(uint8_t rs2, uint8_t rs1, int32_t imm12) const {
                Instr(RvS(static_cast<uint32_t>(imm12), rs2, rs1, F3_SH, STORE));
            }
            void Sw(uint8_t rs2, uint8_t rs1, int32_t imm12) const {
                Instr(RvS(static_cast<uint32_t>(imm12), rs2, rs1, F3_SW, STORE));
            }
            void Sd(uint8_t rs2, uint8_t rs1, int32_t imm12) const {
                Instr(RvS(static_cast<uint32_t>(imm12), rs2, rs1, F3_SD, STORE));
            }

            // Branches (B-type)
            void Beq(uint8_t rs1, uint8_t rs2, int32_t imm13) const {
                Instr(RvB(static_cast<uint32_t>(imm13), rs2, rs1, F3_BEQ, BRANCH));
            }
            void Bne(uint8_t rs1, uint8_t rs2, int32_t imm13) const {
                Instr(RvB(static_cast<uint32_t>(imm13), rs2, rs1, F3_BNE, BRANCH));
            }
            void Blt(uint8_t rs1, uint8_t rs2, int32_t imm13) const {
                Instr(RvB(static_cast<uint32_t>(imm13), rs2, rs1, F3_BLT, BRANCH));
            }
            void Bge(uint8_t rs1, uint8_t rs2, int32_t imm13) const {
                Instr(RvB(static_cast<uint32_t>(imm13), rs2, rs1, F3_BGE, BRANCH));
            }
            void Bltu(uint8_t rs1, uint8_t rs2, int32_t imm13) const {
                Instr(RvB(static_cast<uint32_t>(imm13), rs2, rs1, F3_BLTU, BRANCH));
            }
            void Bgeu(uint8_t rs1, uint8_t rs2, int32_t imm13) const {
                Instr(RvB(static_cast<uint32_t>(imm13), rs2, rs1, F3_BGEU, BRANCH));
            }

            // Jump and link (J-type)
            void Jal(uint8_t rd, int32_t imm21) const {
                Instr(RvJ(static_cast<uint32_t>(imm21), rd, JAL));
            }

            // Jump and link register (I-type)
            void Jalr(uint8_t rd, uint8_t rs1, int32_t imm12) const {
                Instr(RvI(static_cast<uint32_t>(imm12), rs1, 0, rd, JALR));
            }

            // Return: jalr zero, ra, 0
            void Ret() const {
                Jalr(RV_X0, RV_RA, 0);
            }

            // Move (pseudo): addi rd, rs1, 0
            void Mv(uint8_t rd, uint8_t rs1) const {
                Addi(rd, rs1, 0);
            }

            // Neg (pseudo): sub rd, x0, rs1
            void Neg(uint8_t rd, uint8_t rs1) const {
                Sub(rd, RV_X0, rs1);
            }

            // Not (pseudo): xori rd, rs1, -1
            void Not(uint8_t rd, uint8_t rs1) const {
                Xori(rd, rs1, -1);
            }

            // Seqz (pseudo): sltiu rd, rs1, 1
            void Snez(uint8_t rd, uint8_t rs1) const {
                Sltu(rd, RV_X0, rs1);
            }

            // Seqz (pseudo): sltiu rd, rs1, 1
            void Seqz(uint8_t rd, uint8_t rs1) const {
                Sltiu(rd, rs1, 1);
            }

            // Load immediate: handles various sizes
            void Li(uint8_t rd, int64_t val) const {
                uint64_t uval = static_cast<uint64_t>(val);
                // 12-bit signed immediate
                if (static_cast<uint64_t>(val + 2048) <= 4095) {
                    Addi(rd, RV_X0, static_cast<int32_t>(val));
                    return;
                }
                // Check if val is a sign-extended 32-bit value → LUI+ADDIW
                int32_t val32 = static_cast<int32_t>(static_cast<uint32_t>(uval));
                if (static_cast<int64_t>(val32) == val) {
                    int32_t imm12 = val32 & 0xFFF;
                    if (imm12 >= 2048) imm12 -= 4096;
                    int32_t luiPart = val32 - imm12;
                    uint32_t upper20 = static_cast<uint32_t>(
                        static_cast<int32_t>(luiPart >> 12)) & 0xFFFFF;
                    Lui(rd, upper20);
                    if (imm12 != 0) Addiw(rd, rd, imm12);
                    return;
                }
                // Full 64-bit: load hi32 + SLLI + add lo32
                uint32_t hi32 = static_cast<uint32_t>(uval >> 32);
                uint32_t lo32 = static_cast<uint32_t>(uval);
                Li(rd, static_cast<int64_t>(static_cast<int32_t>(hi32)));
                Slli(rd, rd, 32);
                if (lo32 != 0) {
                    Li(RV_T1, static_cast<int64_t>(static_cast<int32_t>(lo32)));
                    // Zero-extend lo32 if upper bits invalid from sign-extension
                    if (lo32 & 0x80000000) {
                        // Upper 32 bits of T1 are 0xFFFFFFFF; clear them
                        Slli(RV_T1, RV_T1, 32);
                        Srli(RV_T1, RV_T1, 32);
                    }
                    Add(rd, rd, RV_T1);
                }
            }

            // Float load/store
            void Fld(uint8_t rd, uint8_t rs1, int32_t imm12) const {
                Instr(RvI(static_cast<uint32_t>(imm12), rs1, F3_FLD, rd, FLOAD));
            }
            void Fsd(uint8_t rs2, uint8_t rs1, int32_t imm12) const {
                Instr(RvS(static_cast<uint32_t>(imm12), rs2, rs1, F3_FSD, FSTORE));
            }
            void Flw(uint8_t rd, uint8_t rs1, int32_t imm12) const {
                Instr(RvI(static_cast<uint32_t>(imm12), rs1, F3_FLW, rd, FLOAD));
            }
            void Fsw(uint8_t rs2, uint8_t rs1, int32_t imm12) const {
                Instr(RvS(static_cast<uint32_t>(imm12), rs2, rs1, F3_FSW, FSTORE));
            }

            // Float ALU (R-type with OP_FP)
            void Fadd_s(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(F7_FADD_S, rs2, rs1, 0, rd, OP_FP));
            }
            void Fsub_s(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(F7_FSUB_S, rs2, rs1, 0, rd, OP_FP));
            }
            void Fmul_s(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(F7_FMUL_S, rs2, rs1, 0, rd, OP_FP));
            }
            void Fdiv_s(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(F7_FDIV_S, rs2, rs1, 0, rd, OP_FP));
            }
            void Feq_s(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(F7_FEQ_S, rs2, rs1, 0, rd, OP_FP));
            }
            void Flt_s(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(F7_FLT_S, rs2, rs1, 0, rd, OP_FP));
            }
            void Fle_s(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(F7_FLE_S, rs2, rs1, 0, rd, OP_FP));
            }

            void Fadd_d(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(F7_FADD_D, rs2, rs1, 0, rd, OP_FP));
            }
            void Fsub_d(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(F7_FSUB_D, rs2, rs1, 0, rd, OP_FP));
            }
            void Fmul_d(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(F7_FMUL_D, rs2, rs1, 0, rd, OP_FP));
            }
            void Fdiv_d(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(F7_FDIV_D, rs2, rs1, 0, rd, OP_FP));
            }
            void Feq_d(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(F7_FEQ_D, rs2, rs1, 0, rd, OP_FP));
            }
            void Flt_d(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(F7_FLT_D, rs2, rs1, 0, rd, OP_FP));
            }
            void Fle_d(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(F7_FLE_D, rs2, rs1, 0, rd, OP_FP));
            }

            // Float sign injection (for negation)
            void Fsgnj_s(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(F7_FSGNJ_S, rs2, rs1, 0, rd, OP_FP));
            }
            void Fsgnjn_s(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(F7_FSGNJ_S, rs2, rs1, 1, rd, OP_FP));
            }
            void Fsgnjx_s(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(F7_FSGNJ_S, rs2, rs1, 2, rd, OP_FP));
            }
            void Fsgnj_d(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(F7_FSGNJ_D, rs2, rs1, 0, rd, OP_FP));
            }
            void Fsgnjn_d(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(F7_FSGNJ_D, rs2, rs1, 1, rd, OP_FP));
            }
            void Fsgnjx_d(uint8_t rd, uint8_t rs1, uint8_t rs2) const {
                Instr(RvR(F7_FSGNJ_D, rs2, rs1, 2, rd, OP_FP));
            }

            // Float convert
            void Fcvt_s_d(uint8_t rd, uint8_t rs1) const {
                Instr(RvR(F7_FCVT_S_D, 0, rs1, 0, rd, OP_FP));
            }
            void Fcvt_d_s(uint8_t rd, uint8_t rs1) const {
                Instr(RvR(F7_FCVT_D_S, 0, rs1, 0, rd, OP_FP));
            }
            void Fcvt_w_s(uint8_t rd, uint8_t rs1) const {
                Instr(RvR(F7_FCVT_W_S, 0, rs1, 0, rd, OP_FP));
            }
            void Fcvt_w_d(uint8_t rd, uint8_t rs1) const {
                Instr(RvR(F7_FCVT_W_D, 0, rs1, 0, rd, OP_FP));
            }
            void Fcvt_l_s(uint8_t rd, uint8_t rs1) const {
                Instr(RvR(F7_FCVT_L_S, 0, rs1, 0, rd, OP_FP));
            }
            void Fcvt_l_d(uint8_t rd, uint8_t rs1) const {
                Instr(RvR(F7_FCVT_L_D, 0, rs1, 0, rd, OP_FP));
            }
            void Fmv_x_w(uint8_t rd, uint8_t rs1) const {
                Instr(RvR(F7_FMV_X_W, 0, rs1, 0, rd, OP_FP));
            }
            void Fmv_x_d(uint8_t rd, uint8_t rs1) const {
                // fmv.x.d: FSGNJ_D rd, rs1, rs1 with funct3=0? No.
                // Actually FSGNJ_D with rs2 = rs1 gives rd = rs1 (no-op in float).
                // fmv.x.d is FCVT_L_D with rounding mode = 0.
                // Actually fmv.x.d is a pseudo for FSGNJ_D rd, rs1, rs1 with
                // funct3=0 which copies the bits (same as fmv.x.d).
                // In RISC-V, fmv.x.d is encoded as FSGNJ_D rd, rs1, rs1 (funct3=0).
                // But FSGNJ_D copies the float value, not the bits. For bitcopy,
                // fmv.x.d is a separate instruction:
                // Actually in RV64, fmv.x.d is FSGNJ_D rd, rs1, rs1.
                // And fmv.d.x is FSGNJ_D rd, rs1, rs1 with rd and rs1 swapped? No.
                // Wait, fmv.x.d is at funct7=0x71, funct3=0, rs2=0, but with
                // a specific rounding mode encoding.
                // Actually for RISC-V: fmv.x.d rs1 is encoded as FSGNJ_D rd, rs1, rs1
                // (yes, FSGNJ_D with both operands same copies the bits to integer).
                Instr(RvR(F7_FSGNJ_D, rs1, rs1, 0, rd, OP_FP));
            }

            // FMV.D.X: move integer to float (bit copy)
            void Fmv_d_x(uint8_t rd, uint8_t rs1) const {
                Instr(RvR(F7_FSGNJ_D, rs1, rs1, 0, rd, OP_FP));
            }

            // FMV.W.X: move integer to float32 (bit copy)
            void Fmv_w_x(uint8_t rd, uint8_t rs1) const {
                Instr(RvR(F7_FSGNJ_S, rs1, rs1, 0, rd, OP_FP));
            }

        private:
            std::vector<uint8_t>& out_;
        };

        // Jump/branch patch info for RISC-V
        struct RiscVJumpPatch {
            uint32_t patchOff;
            uint32_t targetBlock;
            uint8_t kind; // 0 = JAL (J-type), 1 = B-type conditional
            uint8_t rs1, rs2; // registers for B-type
            uint8_t funct3;   // funct3 for B-type
        };

        // RISC-V Code Generator
        class RiscVCodeGen {
        public:
            explicit RiscVCodeGen(
                const LirModule& mod,
                const std::vector<LirStructDecl>& structDecls,
                const std::vector<std::string>& packageInterfaceNames,
                std::string pkgName)
                : mod(mod)
                , structDecls(structDecls)
                , packageInterfaceNames(packageInterfaceNames)
                , pkgName(std::move(pkgName))
                , enc(textData) {}

            RcuFile Generate();

        private:
            const LirModule& mod;
            const std::vector<LirStructDecl>& structDecls;
            const std::vector<std::string>& packageInterfaceNames;
            std::string pkgName;

            std::vector<uint8_t> textData;
            std::vector<uint8_t> rodataData;
            std::vector<uint8_t> dataData;
            std::vector<RcuReloc> textRelocs;
            std::vector<RcuReloc> rodataRelocs;
            std::vector<RcuSymbol> symbols;
            RcuStringTable strings;
            RiscV64Enc enc;

            // Interned rodata constants
            std::unordered_map<std::string, uint32_t> strSyms;
            std::unordered_map<std::string, uint32_t> f32Syms;
            std::unordered_map<std::string, uint32_t> f64Syms;
            int constIdx = 0;
            uint32_t f32SignMaskSym = ~0u;
            uint32_t f64SignMaskSym = ~0u;

            // Extern and func symbol maps
            std::unordered_map<std::string, uint32_t> externSyms;
            std::unordered_map<std::string, uint32_t> funcSyms;
            std::unordered_map<std::string, uint32_t> dataSyms;

            // Per-function state
            struct FieldLayout {
                std::string name;
                int offset = 0;
                int size = 0;
            };
            struct StructLayout {
                std::vector<FieldLayout> fields;
                int totalSize = 0;
                int alignment = 1;
            };
            using LayoutMap = std::unordered_map<std::string, StructLayout>;
            LayoutMap layouts;
            std::unordered_set<std::string> interfaceNames;

            struct PhiMove {
                LirReg dst;
                LirReg src;
                TypeRef type;
            };
            std::unordered_map<LirReg, int32_t> slotMap;
            std::unordered_map<LirReg, int32_t> allocaData;
            std::unordered_map<LirReg, TypeRef> regTypes;
            std::unordered_map<uint32_t,
                std::unordered_map<uint32_t, std::vector<PhiMove>>> phiMoves;
            int32_t nextOff = 0;
            int32_t frameSize = 0;
            int32_t hiddenReturnOff = 0;

            std::vector<uint32_t> blockOffsets;
            std::vector<RiscVJumpPatch> jumpPatches;

            // For jumps that need relocation (cross-function calls)
            struct RelocPatch {
                uint32_t patchOff;
                uint32_t symIdx;
                uint16_t relocType;
            };
            std::vector<RelocPatch> relocPatches;

            [[nodiscard]] int32_t Disp(LirReg r) const {
                // Offset by 16 to skip saved ra (s0-8) and saved s0 (s0-16)
                return -slotMap.at(r) - 16;
            }

            // Ensure an offset from s0 fits in 12-bit signed range.
            // If it does, returns {s0, offset}. Otherwise, computes s0+offset into T1
            // and returns {T1, 0}. Result base in *base, adjusted offset in *adjOff.
            void PrepSOffset(int32_t offset, uint8_t& base, int32_t& adjOff) const {
                if (offset >= -2048 && offset <= 2047) {
                    base = RV_S0;
                    adjOff = offset;
                }
                else {
                    enc.Li(RV_T1, offset);
                    enc.Add(RV_T1, RV_S0, RV_T1);
                    base = RV_T1;
                    adjOff = 0;
                }
            }

            // Shortcut: load from s0+disp into rd, handling large disp
            void LdS0(uint8_t rd, int32_t disp) const {
                uint8_t base;
                int32_t adj;
                PrepSOffset(disp, base, adj);
                enc.Ld(rd, base, adj);
            }
            void LwS0(uint8_t rd, int32_t disp) const {
                uint8_t base;
                int32_t adj;
                PrepSOffset(disp, base, adj);
                enc.Lw(rd, base, adj);
            }
            void LhS0(uint8_t rd, int32_t disp) const {
                uint8_t base;
                int32_t adj;
                PrepSOffset(disp, base, adj);
                enc.Lh(rd, base, adj);
            }
            void LbS0(uint8_t rd, int32_t disp) const {
                uint8_t base;
                int32_t adj;
                PrepSOffset(disp, base, adj);
                enc.Lb(rd, base, adj);
            }
            void LbuS0(uint8_t rd, int32_t disp) const {
                uint8_t base;
                int32_t adj;
                PrepSOffset(disp, base, adj);
                enc.Lbu(rd, base, adj);
            }
            void LhuS0(uint8_t rd, int32_t disp) const {
                uint8_t base;
                int32_t adj;
                PrepSOffset(disp, base, adj);
                enc.Lhu(rd, base, adj);
            }
            void LwuS0(uint8_t rd, int32_t disp) const {
                uint8_t base;
                int32_t adj;
                PrepSOffset(disp, base, adj);
                enc.Lwu(rd, base, adj);
            }
            void SdS0(uint8_t rs2, int32_t disp) const {
                uint8_t base;
                int32_t adj;
                PrepSOffset(disp, base, adj);
                enc.Sd(rs2, base, adj);
            }
            void SwS0(uint8_t rs2, int32_t disp) const {
                uint8_t base;
                int32_t adj;
                PrepSOffset(disp, base, adj);
                enc.Sw(rs2, base, adj);
            }
            void ShS0(uint8_t rs2, int32_t disp) const {
                uint8_t base;
                int32_t adj;
                PrepSOffset(disp, base, adj);
                enc.Sh(rs2, base, adj);
            }
            void SbS0(uint8_t rs2, int32_t disp) const {
                uint8_t base;
                int32_t adj;
                PrepSOffset(disp, base, adj);
                enc.Sb(rs2, base, adj);
            }
            void FlwS0(uint8_t rd, int32_t disp) const {
                uint8_t base;
                int32_t adj;
                PrepSOffset(disp, base, adj);
                enc.Flw(rd, base, adj);
            }
            void FldS0(uint8_t rd, int32_t disp) const {
                uint8_t base;
                int32_t adj;
                PrepSOffset(disp, base, adj);
                enc.Fld(rd, base, adj);
            }
            void FswS0(uint8_t rs2, int32_t disp) const {
                uint8_t base;
                int32_t adj;
                PrepSOffset(disp, base, adj);
                enc.Fsw(rs2, base, adj);
            }
            void FsdS0(uint8_t rs2, int32_t disp) const {
                uint8_t base;
                int32_t adj;
                PrepSOffset(disp, base, adj);
                enc.Fsd(rs2, base, adj);
            }
            void AddiS0(uint8_t rd, int32_t disp) const {
                uint8_t base;
                int32_t adj;
                PrepSOffset(disp, base, adj);
                enc.Addi(rd, base, adj);
            }

            static std::string BaseTypeName(const std::string& name) {
                auto pos = name.find('<');
                return pos == std::string::npos ? name : name.substr(0, pos);
            }

            [[nodiscard]] int SizeOfRuntime(const TypeRef& t) const {
                if (t.kind == TypeRef::Kind::Range) {
                    const TypeRef& elemType =
                        t.inner.empty() ? TypeRef::MakeInt64() : t.inner[0];
                    int elemSize = SizeOf(elemType);
                    return AlignUp(2 * elemSize + 1, elemSize > 0 ? elemSize : 1);
                }
                if (t.kind == TypeRef::Kind::Named) {
                    const std::string base = BaseTypeName(t.name);
                    if (interfaceNames.count(base)) return 16;
                    if (base == "Slice") return 16;
                    auto it = layouts.find(base);
                    if (it != layouts.end()) return it->second.totalSize;
                }
                return SizeOf(t);
            }

            uint32_t AddSymbol(RcuSymbol s) {
                auto idx = static_cast<uint32_t>(symbols.size());
                symbols.push_back(std::move(s));
                return idx;
            }

            uint32_t GetOrAddExtern(const std::string& name,
                                     uint8_t kind,
                                     const std::string& dll = {}) {
                auto it = externSyms.find(name);
                if (it != externSyms.end()) return it->second;
                RcuSymbol s;
                s.name = name;
                s.typeName = dll;
                s.kind = kind;
                s.visibility = RcuSymVis::Global;
                s.sectionIdx = RCU_SEC_EXTERNAL;
                uint32_t idx = AddSymbol(std::move(s));
                externSyms[name] = idx;
                return idx;
            }

            // Intern string/float constants
            uint32_t InternStr(const std::string& s) {
                auto it = strSyms.find(s);
                if (it != strSyms.end()) return it->second;
                AlignRodata(8);
                uint32_t off = static_cast<uint32_t>(rodataData.size());
                RcuSymbol sym;
                sym.name = "__rux_str_" + std::to_string(constIdx++);
                sym.sectionIdx = RCU_RODATA_IDX;
                sym.value = off;
                sym.kind = RcuSymKind::Const;
                sym.visibility = RcuSymVis::Local;
                for (char c : s) rodataData.push_back(c);
                rodataData.push_back(0);
                uint32_t idx = AddSymbol(std::move(sym));
                strSyms[s] = idx;
                return idx;
            }

            uint32_t InternF32(const std::string& s) {
                auto it = f32Syms.find(s);
                if (it != f32Syms.end()) return it->second;
                AlignRodata(4);
                uint32_t off = static_cast<uint32_t>(rodataData.size());
                RcuSymbol sym;
                sym.name = "__rux_f32_" + std::to_string(constIdx++);
                sym.sectionIdx = RCU_RODATA_IDX;
                sym.value = off;
                sym.kind = RcuSymKind::Const;
                sym.visibility = RcuSymVis::Local;
                // Parse float bits
                float f = 0;
                try {
                    f = std::stof(std::string(s));
                } catch (...) {}
                uint32_t bits;
                std::memcpy(&bits, &f, sizeof(bits));
                for (int i = 0; i < 4; ++i) rodataData.push_back((bits >> (i * 8)) & 0xFF);
                uint32_t idx = AddSymbol(std::move(sym));
                f32Syms[s] = idx;
                return idx;
            }

            uint32_t InternF64(const std::string& s) {
                auto it = f64Syms.find(s);
                if (it != f64Syms.end()) return it->second;
                AlignRodata(8);
                uint32_t off = static_cast<uint32_t>(rodataData.size());
                RcuSymbol sym;
                sym.name = "__rux_f64_" + std::to_string(constIdx++);
                sym.sectionIdx = RCU_RODATA_IDX;
                sym.value = off;
                sym.kind = RcuSymKind::Const;
                sym.visibility = RcuSymVis::Local;
                double d = 0;
                try {
                    d = std::stod(std::string(s));
                } catch (...) {}
                uint64_t bits;
                std::memcpy(&bits, &d, sizeof(bits));
                for (int i = 0; i < 8; ++i) rodataData.push_back((bits >> (i * 8)) & 0xFF);
                uint32_t idx = AddSymbol(std::move(sym));
                f64Syms[s] = idx;
                return idx;
            }

            uint32_t InternF32SignMask() {
                if (f32SignMaskSym != ~0u) return f32SignMaskSym;
                AlignRodata(4);
                uint32_t off = static_cast<uint32_t>(rodataData.size());
                RcuSymbol sym;
                sym.name = "__rux_f32_sign";
                sym.sectionIdx = RCU_RODATA_IDX;
                sym.value = off;
                sym.kind = RcuSymKind::Const;
                sym.visibility = RcuSymVis::Global;
                // 0x80000000
                rodataData.push_back(0x00); rodataData.push_back(0x00);
                rodataData.push_back(0x00); rodataData.push_back(0x80);
                f32SignMaskSym = AddSymbol(std::move(sym));
                return f32SignMaskSym;
            }

            uint32_t InternF64SignMask() {
                if (f64SignMaskSym != ~0u) return f64SignMaskSym;
                AlignRodata(8);
                uint32_t off = static_cast<uint32_t>(rodataData.size());
                RcuSymbol sym;
                sym.name = "__rux_f64_sign";
                sym.sectionIdx = RCU_RODATA_IDX;
                sym.value = off;
                sym.kind = RcuSymKind::Const;
                sym.visibility = RcuSymVis::Global;
                // 0x8000000000000000
                for (int i = 0; i < 8; ++i) rodataData.push_back(0);
                rodataData.push_back(0x80);
                f64SignMaskSym = AddSymbol(std::move(sym));
                return f64SignMaskSym;
            }

            void AlignRodata(int a) {
                while (rodataData.size() % static_cast<size_t>(a) != 0) {
                    rodataData.push_back(0);
                }
            }

            void AddTextReloc(uint32_t sectionOff, uint32_t symIdx,
                              int32_t addend = 0) {
                textRelocs.push_back({sectionOff, symIdx, RcuRelType::Rel32, addend});
            }

            void AddRodataReloc(uint32_t sectionOff, uint32_t symIdx,
                                uint16_t type, int32_t addend = 0) {
                rodataRelocs.push_back({sectionOff, symIdx, type, addend});
            }

            // Load value from LIR register to t0 (integer) or ft0 (float)
            void LoadA(LirReg reg, const TypeRef& t) const {
                int sz = SizeOf(t);
                int runtimeSz = SizeOfRuntime(t);
                int32_t d = Disp(reg);
                uint8_t base;
                int32_t adj;
                PrepSOffset(d, base, adj);
                if (runtimeSz == 16) {
                    enc.Ld(RV_T0, base, adj);
                    enc.Ld(RV_T1, base, adj + 8);
                    enc.Mv(RV_A0, RV_T0);
                    // return in a0:a1
                }
                else if (IsFloat(t)) {
                    if (t.kind == TypeRef::Kind::Float32) {
                        enc.Flw(RV_FT0, base, adj);
                    }
                    else {
                        enc.Fld(RV_FT0, base, adj);
                    }
                }
                else if (sz == 1 && !t.IsSigned()) {
                    enc.Lbu(RV_T0, base, adj);
                }
                else if (sz == 1) {
                    enc.Lb(RV_T0, base, adj);
                }
                else if (sz == 2 && !t.IsSigned()) {
                    enc.Lhu(RV_T0, base, adj);
                }
                else if (sz == 2) {
                    enc.Lh(RV_T0, base, adj);
                }
                else if (sz == 4 && !t.IsSigned()) {
                    enc.Lwu(RV_T0, base, adj);
                }
                else if (sz == 4) {
                    enc.Lw(RV_T0, base, adj);
                }
                else {
                    enc.Ld(RV_T0, base, adj);
                }
            }

            // Load into t1 (second operand)
            void LoadB(LirReg reg, const TypeRef& t) const {
                int sz = SizeOf(t);
                int32_t d = Disp(reg);
                uint8_t base;
                int32_t adj;
                PrepSOffset(d, base, adj);
                if (IsFloat(t)) {
                    if (t.kind == TypeRef::Kind::Float32) {
                        enc.Flw(1, base, adj); // ft1
                    }
                    else {
                        enc.Fld(1, base, adj); // ft1
                    }
                }
                else if (sz == 0 || sz == 8) {
                    enc.Ld(RV_T1, base, adj);
                }
                else if (sz == 1 && !t.IsSigned()) {
                    enc.Lbu(RV_T1, base, adj);
                }
                else if (sz == 1) {
                    enc.Lb(RV_T1, base, adj);
                }
                else if (sz == 2 && !t.IsSigned()) {
                    enc.Lhu(RV_T1, base, adj);
                }
                else if (sz == 2) {
                    enc.Lh(RV_T1, base, adj);
                }
                else if (sz == 4 && !t.IsSigned()) {
                    enc.Lwu(RV_T1, base, adj);
                }
                else {
                    enc.Lw(RV_T1, base, adj);
                }
            }

            void StoreA(LirReg dst, const TypeRef& t) const {
                int sz = SizeOf(t);
                int runtimeSz = SizeOfRuntime(t);
                int32_t d = Disp(dst);
                uint8_t base;
                int32_t adj;
                PrepSOffset(d, base, adj);
                if (runtimeSz == 16) {
                    enc.Sd(RV_T0, base, adj);
                    enc.Sd(RV_T1, base, adj + 8);
                }
                else if (IsFloat(t)) {
                    if (t.kind == TypeRef::Kind::Float32) {
                        enc.Fsw(RV_FT0, base, adj);
                    }
                    else {
                        enc.Fsd(RV_FT0, base, adj);
                    }
                }
                else if (sz == 0 || sz == 8) {
                    enc.Sd(RV_T0, base, adj);
                }
                else if (sz == 4) {
                    enc.Sw(RV_T0, base, adj);
                }
                else if (sz == 2) {
                    enc.Sh(RV_T0, base, adj);
                }
                else {
                    enc.Sb(RV_T0, base, adj);
                }
            }

            void LoadReturnValue(LirReg reg, const TypeRef& t) const {
                int runtimeSz = SizeOfRuntime(t);
                int32_t d = Disp(reg);
                uint8_t base;
                int32_t adj;
                PrepSOffset(d, base, adj);
                if (runtimeSz == 16) {
                    enc.Ld(RV_A0, base, adj);
                    enc.Ld(RV_A1, base, adj + 8);
                    return;
                }
                int sz = SizeOf(t);
                if (IsFloat(t)) {
                    if (t.kind == TypeRef::Kind::Float32) {
                        enc.Flw(RV_FA0, base, adj);
                    }
                    else {
                        enc.Fld(RV_FA0, base, adj);
                    }
                }
                else if (sz == 1 && !t.IsSigned()) {
                    enc.Lbu(RV_A0, base, adj);
                }
                else if (sz == 1) {
                    enc.Lb(RV_A0, base, adj);
                }
                else if (sz == 2 && !t.IsSigned()) {
                    enc.Lhu(RV_A0, base, adj);
                }
                else if (sz == 2) {
                    enc.Lh(RV_A0, base, adj);
                }
                else if (sz == 4 && !t.IsSigned()) {
                    enc.Lwu(RV_A0, base, adj);
                }
                else if (sz == 4) {
                    enc.Lw(RV_A0, base, adj);
                }
                else {
                    enc.Ld(RV_A0, base, adj);
                }
            }

            void StoreReturnValue(LirReg dst, const TypeRef& t) const {
                // Result is in a0 (int) or fa0 (float)
                int runtimeSz = SizeOfRuntime(t);
                int32_t d = Disp(dst);
                uint8_t base;
                int32_t adj;
                PrepSOffset(d, base, adj);
                if (runtimeSz == 16) {
                    enc.Sd(RV_A0, base, adj);
                    enc.Sd(RV_A1, base, adj + 8);
                    return;
                }
                int sz = SizeOf(t);
                if (IsFloat(t)) {
                    if (t.kind == TypeRef::Kind::Float32) {
                        enc.Fsw(RV_FA0, base, adj);
                    }
                    else {
                        enc.Fsd(RV_FA0, base, adj);
                    }
                }
                else if (sz == 0 || sz == 8) {
                    enc.Sd(RV_A0, base, adj);
                }
                else if (sz == 4) {
                    enc.Sw(RV_A0, base, adj);
                }
                else if (sz == 2) {
                    enc.Sh(RV_A0, base, adj);
                }
                else {
                    enc.Sb(RV_A0, base, adj);
                }
            }

            void StoreHiddenReturnValue(LirReg src, const TypeRef& t) const {
                if (hiddenReturnOff == 0 || SizeOfRuntime(t) != 16) {
                    LoadReturnValue(src, t);
                    return;
                }
                int32_t hOff = -hiddenReturnOff;
                uint8_t hBase;
                int32_t hAdj;
                PrepSOffset(hOff, hBase, hAdj);
                enc.Ld(RV_T2, hBase, hAdj); // load hidden ptr
                int32_t sOff = Disp(src);
                uint8_t sBase;
                int32_t sAdj;
                PrepSOffset(sOff, sBase, sAdj);
                enc.Ld(RV_T0, sBase, sAdj);
                enc.Sd(RV_T0, RV_T2, 0);
                enc.Ld(RV_T0, sBase, sAdj + 8);
                enc.Sd(RV_T0, RV_T2, 8);
                enc.Mv(RV_A0, RV_T2); // return hidden ptr
            }

            // Slot allocation
            int32_t AllocSlot(LirReg reg, int bytes) {
                if (auto it = slotMap.find(reg); it != slotMap.end()) {
                    return it->second;
                }
                int slotBytes = std::max(bytes, 8);
                int al = std::min(slotBytes, 8);
                nextOff = AlignUp(nextOff, al);
                nextOff += slotBytes;
                slotMap[reg] = nextOff;
                return nextOff;
            }

            int32_t AllocRegion(int bytes) {
                int al = (bytes > 0) ? std::min(bytes, 8) : 1;
                nextOff = AlignUp(nextOff, al);
                nextOff += (bytes > 0 ? bytes : 8);
                return nextOff;
            }

            int FieldOffset(LirReg base, const std::string& fieldName) {
                auto typeIt = regTypes.find(base);
                if (typeIt == regTypes.end()) return 0;
                const TypeRef& pt = typeIt->second;
                const TypeRef& inner = (pt.kind == TypeRef::Kind::Pointer && !pt.inner.empty())
                    ? pt.inner[0] : pt;
                if (inner.kind == TypeRef::Kind::Range) {
                    const TypeRef& elemType = inner.inner.empty()
                        ? TypeRef::MakeInt64() : inner.inner[0];
                    int elemSize = SizeOf(elemType);
                    if (fieldName == "lo") return 0;
                    if (fieldName == "hi") return elemSize;
                    if (fieldName == "inclusive") return 2 * elemSize;
                    return 0;
                }
                if (inner.kind == TypeRef::Kind::Tuple) {
                    std::size_t idx = 0;
                    try { idx = std::stoul(fieldName); } catch (...) { return 0; }
                    int offset = 0;
                    if (idx >= inner.inner.size()) return 0;
                    for (std::size_t i = 0; i < idx && i < inner.inner.size(); ++i) {
                        const int sz = SizeOf(inner.inner[i]);
                        const int al = sz > 0 ? std::min(sz, 8) : 1;
                        if (al > 1) offset = AlignUp(offset, al);
                        offset += sz > 0 ? sz : 8;
                    }
                    const int fieldSize = SizeOf(inner.inner[idx]);
                    const int fieldAlign = fieldSize > 0 ? std::min(fieldSize, 8) : 1;
                    if (fieldAlign > 1) offset = AlignUp(offset, fieldAlign);
                    return offset;
                }
                if (inner.kind != TypeRef::Kind::Named) return 0;
                const std::string baseName = BaseTypeName(inner.name);
                if (interfaceNames.count(baseName)) {
                    if (fieldName == "data") return 0;
                    if (fieldName == "vtable") return 8;
                    return 0;
                }
                if (baseName == "Slice") {
                    if (fieldName == "data") return 0;
                    if (fieldName == "length") return 8;
                    return 0;
                }
                auto layIt = layouts.find(baseName);
                if (layIt == layouts.end()) {
                    if (baseName.find("String") != std::string::npos || baseName.find("length") != std::string::npos || fieldName == "length")
                        fprintf(stderr, "  FIELD_OFFSET baseName=%s field=%s NOT FOUND in layouts\n", baseName.c_str(), fieldName.c_str());
                    return 0;
                }
                if (baseName.find("String") != std::string::npos || fieldName == "length")
                    fprintf(stderr, "  FIELD_OFFSET baseName=%s field=%s -> looking up in layouts\n", baseName.c_str(), fieldName.c_str());
                for (const auto& f : layIt->second.fields) {
                    if (f.name == fieldName) {
                        if (baseName.find("String") != std::string::npos || fieldName == "length")
                            fprintf(stderr, "  FIELD_OFFSET found %s at offset %d\n", fieldName.c_str(), f.offset);
                        return f.offset;
                    }
                }
                return 0;
            }

            void BuildLayouts() {
                for (const auto& name : packageInterfaceNames) {
                    interfaceNames.insert(name);
                }
                for (const auto& s : structDecls) {
                    StructLayout layout;
                    int offset = 0, maxAlign = 1;
                    for (const auto& f : s.fields) {
                        int sz = SizeOfRuntime(f.type);
                        int al = sz > 0 ? std::min(sz, 8) : 1;
                        if (f.type.kind == TypeRef::Kind::Named) {
                            auto it = layouts.find(BaseTypeName(f.type.name));
                            if (it != layouts.end()) {
                                sz = it->second.totalSize;
                                al = it->second.alignment;
                            }
                        }
                        if (al > 1) offset = AlignUp(offset, al);
                        layout.fields.push_back({f.name, offset, sz});
                        offset += (sz > 0 ? sz : 8);
                        maxAlign = std::max(maxAlign, al);
                    }
                    layout.totalSize = AlignUp(offset, maxAlign);
                    layout.alignment = maxAlign;
                    if (s.name.find("String") != std::string::npos || s.name.find("Slice") != std::string::npos) {
                        fprintf(stderr, "  LAYOUT struct=%s totalSize=%d\n", s.name.c_str(), layout.totalSize);
                        for (auto& f : layout.fields)
                            fprintf(stderr, "    field=%s offset=%d size=%d\n", f.name.c_str(), f.offset, f.size);
                    }
                    layouts[s.name] = std::move(layout);
                }
            }

            void PredeclareFunctions() {
                for (const auto& func : mod.funcs) {
                    if (!func.name.empty() && !func.isExtern) {
                        RcuSymbol s;
                        s.name = func.name;
                        s.sectionIdx = RCU_TEXT_IDX;
                        s.value = 0;
                        s.kind = RcuSymKind::Func;
                        s.visibility =
                            func.isPublic ? RcuSymVis::Global : RcuSymVis::Local;
                        s.typeName = func.returnType.ToString();
                        funcSyms[func.name] = AddSymbol(std::move(s));
                    }
                }
            }

            bool HasPhiMoves(uint32_t from, uint32_t to) const {
                auto it = phiMoves.find(from);
                return it != phiMoves.end() && it->second.contains(to);
            }

            void EmitPhiMoves(uint32_t from, uint32_t to) {
                auto it1 = phiMoves.find(from);
                if (it1 == phiMoves.end()) return;
                auto it2 = it1->second.find(to);
                if (it2 == it1->second.end()) return;
                for (const auto& m : it2->second) {
                    if (!slotMap.contains(m.src)) continue;
                    LoadA(m.src, m.type);
                    StoreA(m.dst, m.type);
                }
            }

            void PrepassFunc(const LirFunc& func) {
                nextOff = 0;
                frameSize = 0;
                hiddenReturnOff = 0;
                slotMap.clear();
                allocaData.clear();
                regTypes.clear();
                phiMoves.clear();
                // No Win64-by-ref on RISC-V; args passed directly in a0-a7.
                // RISC-V lp64d: 16-byte returns use hidden pointer in a0.
                // Allocate a slot for a0 AFTER regular slots so it doesn't
                // collide with parameters.
                for (const auto& p : func.params) {
                    int sz = SizeOfRuntime(p.type);
                    AllocSlot(p.reg, sz > 0 ? sz : 8);
                    regTypes[p.reg] = p.type;
                }
                struct PendingAlloca { LirReg dst; int dsz; };
                std::vector<PendingAlloca> pendingAllocas;
                for (uint32_t bi = 0; bi < func.blocks.size(); ++bi) {
                    for (const auto& instr : func.blocks[bi].instrs) {
                        if (instr.op == LirOpcode::Phi) {
                            for (const auto& [src, pred] : instr.phiPreds) {
                                phiMoves[pred][bi].push_back(
                                    {instr.dst, src, instr.type});
                            }
                        }
                        if (instr.dst == LirNoReg) continue;
                        if (instr.op == LirOpcode::Alloca) {
                            int dsz;
                            if (!instr.strArg.empty()) {
                                int count = 0;
                                try { count = std::stoi(instr.strArg); }
                                catch (...) {}
                                const TypeRef& elemType = instr.type.inner.empty()
                                    ? instr.type : instr.type.inner[0];
                                int elemSize = SizeOfRuntime(elemType);
                                dsz = count * (elemSize > 0 ? elemSize : 8);
                            }
                            else {
                                dsz = SizeOfRuntime(instr.type);
                            }
                            pendingAllocas.push_back({instr.dst, dsz});
                            AllocSlot(instr.dst, 8);
                            regTypes[instr.dst] = TypeRef::MakePointer(instr.type);
                        }
                        else {
                            int sz = SizeOfRuntime(instr.type);
                            if (instr.type.ToString().find("String") != std::string::npos && func.name.find("Main") != std::string::npos ||
                                instr.type.ToString().find("String") != std::string::npos && func.name.find("+__String") != std::string::npos) {
                                fprintf(stderr, "  [SZ] func=%s dst=r%d type=%s sz=%d\n",
                                        func.name.c_str(), instr.dst, instr.type.ToString().c_str(), sz);
                            }
                            AllocSlot(instr.dst, sz > 0 ? sz : 8);
                            regTypes[instr.dst] = instr.type;
                        }
                    }
                }
                // Allocate all alloca regions AFTER all slots, at the bottom of
                // the frame, so that writes to one alloca's region cannot
                // overwrite another alloca's pointer slot or region.
                if (func.name.find("Main") != std::string::npos ||
                    func.name.find("__String") != std::string::npos ||
                    func.name.find("ToString") != std::string::npos ||
                    func.name.find("Print") != std::string::npos ||
                    func.name.find("String::From") != std::string::npos) {
                    fprintf(stderr, "  %s: nextOff before regions = %d\n", func.name.c_str(), nextOff);
                    for (auto& pa : pendingAllocas) {
                        int n0 = nextOff;
                        allocaData[pa.dst] = AllocRegion(pa.dsz > 0 ? pa.dsz : 8);
                        fprintf(stderr, "  AllocRegion(r%d, dsz=%d): nextOff %d -> %d, dataOff=%d\n",
                                pa.dst, pa.dsz, n0, nextOff, allocaData[pa.dst]);
                    }
                } else if (!pendingAllocas.empty()) {
                    fprintf(stderr, "  %s: %zu pendingAllocas\n", func.name.c_str(), pendingAllocas.size());
                    for (auto& pa : pendingAllocas)
                        allocaData[pa.dst] = AllocRegion(pa.dsz > 0 ? pa.dsz : 8);
                }
                frameSize = AlignUp(nextOff, 16);
                if (SizeOfRuntime(func.returnType) == 16) {
                    // Hidden return pointer slot must be past all register slots.
                    // Each register slot at slotMap[r] maps to s0 - slotMap[r] - 16.
                    // The last slot is at s0 - nextOff - 16. We place the hidden
                    // return pointer 8 bytes below that: s0 - nextOff - 24.
                    nextOff = AlignUp(nextOff, 8);
                    hiddenReturnOff = nextOff + 24;
                    nextOff += 8;
                    frameSize = AlignUp(nextOff, 16);
                }
                if (frameSize == 0) frameSize = 16;
            }

            void PatchJumps() {
                for (const auto& p : jumpPatches) {
                    auto target = static_cast<int32_t>(blockOffsets[p.targetBlock]);
                    int32_t disp = target - static_cast<int32_t>(p.patchOff);
                    if (p.kind == 0) {
                        // JAL (J-type)
                        enc.PatchAt(p.patchOff, RvJ(static_cast<uint32_t>(disp),
                                                     RV_X0, JAL));
                    }
                    else {
                        // B-type conditional
                        enc.PatchAt(p.patchOff,
                                    RvB(static_cast<uint32_t>(disp),
                                        p.rs2, p.rs1, p.funct3, BRANCH));
                    }
                }
                jumpPatches.clear();
            }

            void EmitStackAlloc(int32_t bytes) {
                constexpr int32_t kPageSize = 4096;
                while (bytes > kPageSize) {
                    enc.Addi(RV_SP, RV_SP, -kPageSize);
                    // Touch stack for page probe
                    enc.Sd(RV_X0, RV_SP, 0);
                    bytes -= kPageSize;
                }
                if (bytes > 0) {
                    enc.Addi(RV_SP, RV_SP, -bytes);
                }
            }

void EmitCallArgs(const std::vector<LirReg>& args,
                          CallingConvention conv = CallingConvention::Default,
                          int startIdx = 0) {
                // RISC-V lp64d calling convention:
                // int args in a0-a7 (8 regs), float args in fa0-fa7
                int intIdx = startIdx, fltIdx = startIdx;
                for (LirReg arg : args) {
                    TypeRef at = regTypes.contains(arg)
                        ? regTypes.at(arg) : TypeRef::MakeInt64();
                    int32_t d = Disp(arg);
                    uint8_t base;
                    int32_t adj;
                    PrepSOffset(d, base, adj);
                    if (IsFloat(at)) {
                        if (fltIdx < 8) {
                            if (SizeOf(at) == 4) {
                                enc.Flw(RV_FA0 + fltIdx, base, adj);
                            }
                            else {
                                enc.Fld(RV_FA0 + fltIdx, base, adj);
                            }
                            ++fltIdx;
                        }
                    }
                    else {
                        int argSz = SizeOfRuntime(at);
                        // Check if this is an Alloca register with a data region
                        // If yes, and the argument size is 8 (pointer), but the data region
                        // holds a struct value (like String), load 16 bytes
                        bool hasDataRegion = allocaData.contains(arg);
                        bool isPointerType = (at.kind == TypeRef::Kind::Pointer);
                        // For struct types (like String), the register type is *String
                        // but the actual value is 16 bytes stored in the data region
                        if (hasDataRegion && isPointerType) {
                            // For RISC-V, if this register holds a String value (16 bytes)
                            // in its data region, pass the full 16 bytes
                            // Check if the data region size is 16 (String)
                            auto it = allocaData.find(arg);
                            if (it != allocaData.end()) {
                                int dsz = it->second;
                                if (dsz == 16) {
                                    if (intIdx < 8) {
                                        enc.Ld(RV_A0 + intIdx, base, adj);
                                        if (intIdx + 1 < 8) {
                                            enc.Ld(RV_A0 + intIdx + 1, base, adj + 8);
                                        }
                                        intIdx += 2;
                                    }
                                    continue;
                                }
                            }
                        }
                        if (intIdx < 8) {
                            if (argSz == 16) {
                                enc.Ld(RV_A0 + intIdx, base, adj);
                                if (intIdx + 1 < 8) {
                                    enc.Ld(RV_A0 + intIdx + 1, base, adj + 8);
                                }
                                intIdx += 2;
                            }
                            else {
                                enc.Ld(RV_A0 + intIdx, base, adj);
                                ++intIdx;
                            }
                        }
                    }
                }
            }

            void GenInstr(const LirInstr& instr) {
                switch (instr.op) {
                case LirOpcode::Const: {
                    if (instr.dst == LirNoReg) break;
                    const TypeRef& t = instr.type;
                    int sz = SizeOf(t);
                    if (t.kind == TypeRef::Kind::Str) {
                        uint32_t symIdx = InternStr(instr.strArg);
                        // AUIPC + ADDI to get address of string
                        uint32_t pcOff = enc.Size();
                        // Emit AUIPC placeholder + ADDI placeholder
                        enc.Auipc(RV_T0, 0);
                        uint32_t auipcOff = enc.Size();
                        // ADDI rd, rs1, 0 -- placeholder
                        // Actually we need the full AUIPC + ADDI pair.
                        // The AUIPC gives upper 20 bits, ADDI lower 12.
                        // We'll emit AUIPC + ADDI and the linker will patch both.
                        // For now: just AUIPC + ADDI with Rel32 reloc
                        enc.Addi(RV_T0, RV_T0, 0);
                        uint32_t addiOff = enc.Size();
                        // We'll use two Rel32-style relocs. But easier: store
                        // the combined offset as a single AUIPC instruction
                        // and patch the ADDI too.
                        // Actually, let's use a simpler approach: emit
                        // LUI+ADDI and use the linker to resolve the absolute addr.
                        // We don't know the final address, so we need a relocation.
                        // Simplest: use AUIPC+ADDI with a pair reloc.
                        // For now: just emit a placeholder and add reloc.
                        // The relocation will point at the AUIPC and we'll
                        // also store info to patch ADDI.
                        // Let's just use the standard approach:
                        // Emit: auipc t0, 0   (with R_RISCV_PCREL_HI20 reloc)
                        //       addi  t0, t0, 0 (with R_RISCV_PCREL_LO12_I reloc)
                        auto symIdx2 = symIdx; // use the string symbol
                        // Actually for internal strings in rodata, we need
                        // PC-relative addressing. Let's use a simpler method:
                        // just AUIPC + ADDI + add a record for patching.
                        // The linker will need to know about this pair.
                        // For now: we already have a working approach:
                        // Just add the AUIPC offset and ADDI offset for the linker.
                        // But the current Linker doesn't know about RISC-V relocs.
                        // So we need to handle this in the linker.
                        // For now: emit both and mark the AUIPC with a Rel32.
                        // We'll use the AUIPC offset as the reloc site.
                        // The linker will patch the AUIPC with the full PC-relative
                        // offset, and we'll manually patch the ADDI.
                        // But that won't work because AUIPC + ADDI need cooperation.
                        // Alternative: use a single LUI+ADDI with HI20+LO12.
                        // The LUI gets the upper 20 bits, ADDI the lower 12.
                        // We need a relocation for LUI (R_RISCV_HI20) and for ADDI
                        // (R_RISCV_LO12_I). These are paired.
                        // Or we could use AUIPC + LD (for data pointer loads).
                        // Simplest for now: emit LUI + ADDI + mark both with relocs.
                        // But the existing Linker doesn't know about these.
                        // Let me handle it differently:
                        // Emit:
                        //   auipc t0, 0
                        //   addi t0, t0, 0
                        // Store the AUIPC offset and ADDI offset.
                        // After linking, we know the final address.
                        // Then we can compute for AUIPC: (addr - pc) >> 12
                        // and for ADDI: low 12 bits of (addr - pc).
                        // For now, mark the AUIPC with a special reloc.
                        // We'll store both offsets in the relocation.
                        RelocPatch rp;
                        rp.patchOff = pcOff;
                        rp.symIdx = symIdx;
                        rp.relocType = 100; // special: AUIPC+ADDI pair
                        relocPatches.push_back(rp);
                        StoreA(instr.dst, t);
                    }
                    else if (t.kind == TypeRef::Kind::Float32) {
                        uint32_t symIdx = InternF32(instr.strArg);
                        // Load float from rodata via AUIPC + FLD
                        uint32_t pcOff = enc.Size();
                        enc.Auipc(RV_T0, 0);
                        uint32_t auipcEnd = enc.Size();
                        enc.Fld(RV_FT0, RV_T0, 0);
                        RelocPatch rp;
                        rp.patchOff = pcOff;
                        rp.symIdx = symIdx;
                        rp.relocType = 101; // AUIPC+FLD pair
                        relocPatches.push_back(rp);
                        StoreA(instr.dst, t);
                    }
                    else if (t.kind == TypeRef::Kind::Float64) {
                        uint32_t symIdx = InternF64(instr.strArg);
                        uint32_t pcOff = enc.Size();
                        enc.Auipc(RV_T0, 0);
                        uint32_t auipcEnd = enc.Size();
                        enc.Fld(RV_FT0, RV_T0, 0);
                        RelocPatch rp;
                        rp.patchOff = pcOff;
                        rp.symIdx = symIdx;
                        rp.relocType = 101; // AUIPC+FLD pair
                        relocPatches.push_back(rp);
                        StoreA(instr.dst, t);
                    }
                    else if (t.IsBool()) {
                        enc.Li(RV_T0, (instr.strArg == "true" || instr.strArg == "1") ? 1 : 0);
                        StoreA(instr.dst, t);
                    }
                    else {
                        const std::string& sv = instr.strArg.empty() ? "0" : instr.strArg;
                        const std::uint64_t bits = ParseIntegerLiteralBits(sv).value_or(0);
                        enc.Li(RV_T0, static_cast<int64_t>(bits));
                        StoreA(instr.dst, sz > 0 ? t : TypeRef::MakeInt64());
                    }
                    break;
                }
                case LirOpcode::Alloca: {
                    int32_t dataOff = allocaData.at(instr.dst);
                    // Compute stack address: s0 - dataOff - 16
                    // Offset by 16 to match Disp() — skip saved ra (s0-8) and s0 (s0-16)
                    enc.Li(RV_T0, -dataOff - 16);
                    enc.Add(RV_T0, RV_S0, RV_T0);
                    SdS0(RV_T0, Disp(instr.dst));
                    break;
                }
                case LirOpcode::Load: {
                    const TypeRef& t = instr.type;
                    int sz = SizeOf(t);
                    int runtimeSz = SizeOfRuntime(t);
                    if (!instr.strArg.empty()) {
                        // Load from global via AUIPC + LD
                        uint32_t symIdx = GetOrAddExtern(
                            instr.strArg, RcuSymKind::ExternData);
                        uint32_t pcOff = enc.Size();
                        enc.Auipc(RV_T0, 0);
                        enc.Ld(RV_T0, RV_T0, 0);
                        RelocPatch rp;
                        rp.patchOff = pcOff;
                        rp.symIdx = symIdx;
                        rp.relocType = 102; // AUIPC+LD pair
                        relocPatches.push_back(rp);
                        StoreA(instr.dst, sz > 0 ? t : TypeRef::MakeInt64());
                        break;
                    }
                    LirReg ptr = instr.srcs[0];
                    LdS0(RV_T2, Disp(ptr)); // load pointer into t2
                    if (runtimeSz == 16) {
                        enc.Ld(RV_T0, RV_T2, 0);
                        enc.Ld(RV_T1, RV_T2, 8);
                        StoreA(instr.dst, t);
                        break;
                    }
                    if (IsFloat(t)) {
                        if (sz == 4) {
                            enc.Flw(RV_FT0, RV_T2, 0);
                        }
                        else {
                            enc.Fld(RV_FT0, RV_T2, 0);
                        }
                    }
                    else {
                        if (sz == 8 || sz == 0) {
                            enc.Ld(RV_T0, RV_T2, 0);
                        }
                        else if (t.IsSigned()) {
                            if (sz == 4) enc.Lw(RV_T0, RV_T2, 0);
                            else if (sz == 2) enc.Lh(RV_T0, RV_T2, 0);
                            else enc.Lb(RV_T0, RV_T2, 0);
                        }
                        else {
                            if (sz == 4) enc.Lwu(RV_T0, RV_T2, 0);
                            else if (sz == 2) enc.Lhu(RV_T0, RV_T2, 0);
                            else enc.Lbu(RV_T0, RV_T2, 0);
                        }
                    }
                    StoreA(instr.dst, sz > 0 ? t : TypeRef::MakeInt64());
                    break;
                }
                case LirOpcode::Store: {
                    LirReg val = instr.srcs[0];
                    LirReg ptr = instr.srcs[1];
                    const TypeRef& t = instr.type;
                    int sz = SizeOf(t);
                    int runtimeSz = SizeOfRuntime(t);
                    LdS0(RV_T2, Disp(ptr)); // pointer into t2
                    if (runtimeSz == 16) {
                        int32_t vOff = Disp(val);
                        uint8_t vBase;
                        int32_t vAdj;
                        PrepSOffset(vOff, vBase, vAdj);
                        enc.Ld(RV_T0, vBase, vAdj);
                        enc.Sd(RV_T0, RV_T2, 0);
                        enc.Ld(RV_T0, vBase, vAdj + 8);
                        enc.Sd(RV_T0, RV_T2, 8);
                        break;
                    }
                    if (IsFloat(t)) {
                        LoadA(val, t);
                        if (sz == 4) {
                            enc.Fsw(RV_FT0, RV_T2, 0);
                        }
                        else {
                            enc.Fsd(RV_FT0, RV_T2, 0);
                        }
                    }
                    else {
                        const int ss = (sz > 0) ? sz : 8;
                        LoadA(val, t);
                        if (ss == 8) enc.Sd(RV_T0, RV_T2, 0);
                        else if (ss == 4) enc.Sw(RV_T0, RV_T2, 0);
                        else if (ss == 2) enc.Sh(RV_T0, RV_T2, 0);
                        else enc.Sb(RV_T0, RV_T2, 0);
                    }
                    break;
                }
                case LirOpcode::Add:
                case LirOpcode::Sub:
                case LirOpcode::And:
                case LirOpcode::Or:
                case LirOpcode::Xor: {
                    const TypeRef& t = instr.type;
                    if (IsFloat(t)) {
                        LoadA(instr.srcs[0], t);
                        LoadB(instr.srcs[1], t);
                        bool f32 = (t.kind == TypeRef::Kind::Float32);
                        if (instr.op == LirOpcode::Add) {
                            f32 ? enc.Fadd_s(RV_FT0, RV_FT0, 1)
                                : enc.Fadd_d(RV_FT0, RV_FT0, 1);
                        }
                        else if (instr.op == LirOpcode::Sub) {
                            f32 ? enc.Fsub_s(RV_FT0, RV_FT0, 1)
                                : enc.Fsub_d(RV_FT0, RV_FT0, 1);
                        }
                        else {
                            // bitwise on float: use fsgnj for xor
                            // Actually just do integer ALU
                            // For And/Or/Xor on floats, treat as int
                            // We'll load as int and do integer ops
                            // (not commonly used but handle it)
                            LoadA(instr.srcs[0], TypeRef::MakeUInt64());
                            LoadB(instr.srcs[1], TypeRef::MakeUInt64());
                            if (instr.op == LirOpcode::And)
                                enc.And_(RV_T0, RV_T0, RV_T1);
                            else if (instr.op == LirOpcode::Or)
                                enc.Or_(RV_T0, RV_T0, RV_T1);
                            else
                                enc.Xor_(RV_T0, RV_T0, RV_T1);
                        }
                        StoreA(instr.dst, t);
                    }
                    else {
                        LoadA(instr.srcs[0], t);
                        LoadB(instr.srcs[1], t);
                        if (instr.op == LirOpcode::Add)
                            enc.Add(RV_T0, RV_T0, RV_T1);
                        else if (instr.op == LirOpcode::Sub)
                            enc.Sub(RV_T0, RV_T0, RV_T1);
                        else if (instr.op == LirOpcode::And)
                            enc.And_(RV_T0, RV_T0, RV_T1);
                        else if (instr.op == LirOpcode::Or)
                            enc.Or_(RV_T0, RV_T0, RV_T1);
                        else
                            enc.Xor_(RV_T0, RV_T0, RV_T1);
                        StoreA(instr.dst, t);
                    }
                    break;
                }
                case LirOpcode::Mul: {
                    const TypeRef& t = instr.type;
                    if (IsFloat(t)) {
                        LoadA(instr.srcs[0], t);
                        LoadB(instr.srcs[1], t);
                        if (t.kind == TypeRef::Kind::Float32)
                            enc.Fmul_s(RV_FT0, RV_FT0, 1);
                        else
                            enc.Fmul_d(RV_FT0, RV_FT0, 1);
                    }
                    else {
                        LoadA(instr.srcs[0], t);
                        LoadB(instr.srcs[1], t);
                        enc.Mul(RV_T0, RV_T0, RV_T1);
                    }
                    StoreA(instr.dst, t);
                    break;
                }
                case LirOpcode::Div:
                case LirOpcode::Mod: {
                    const TypeRef& t = instr.type;
                    if (IsFloat(t)) {
                        LoadA(instr.srcs[0], t);
                        LoadB(instr.srcs[1], t);
                        if (instr.op == LirOpcode::Div) {
                            if (t.kind == TypeRef::Kind::Float32)
                                enc.Fdiv_s(RV_FT0, RV_FT0, 1);
                            else
                                enc.Fdiv_d(RV_FT0, RV_FT0, 1);
                        }
                        else {
                            // Float mod: call fmodf / fmod
                            uint32_t sym = GetOrAddExtern(t.kind == TypeRef::Kind::Float32 ? "fmodf" : "fmod", RcuSymKind::ExternFunc);
                            EmitCallArgs(instr.srcs, instr.callConv);
                            uint32_t pcOff = enc.Size();
                            enc.Auipc(RV_RA, 0);
                            enc.Jalr(RV_RA, RV_RA, 0);
                            RelocPatch rp;
                            rp.patchOff = pcOff;
                            rp.symIdx = sym;
                            rp.relocType = 103;
                            relocPatches.push_back(rp);
                            if (t.kind == TypeRef::Kind::Float32)
                                enc.Fmv_w_x(RV_FT0, RV_A0);
                            else
                                enc.Fmv_d_x(RV_FT0, RV_A0);
                        }
                        StoreA(instr.dst, t);
                    }
                    else {
                        LoadA(instr.srcs[0], t);
                        LoadB(instr.srcs[1], t);
                        // Move src1 (divisor) to t2 because div clobbers
                        // rs1/rd
                        enc.Mv(RV_T2, RV_T1);
                        if (t.IsSigned()) {
                            enc.Div(RV_T0, RV_T0, RV_T2);
                            // For signed mod, we need rem
                            if (instr.op == LirOpcode::Mod) {
                                // Re-load and compute rem
                                LoadA(instr.srcs[0], t);
                                LoadB(instr.srcs[1], t);
                                enc.Rem(RV_T0, RV_T0, RV_T2);
                            }
                        }
                        else {
                            enc.Divu(RV_T0, RV_T0, RV_T2);
                            if (instr.op == LirOpcode::Mod) {
                                LoadA(instr.srcs[0], t);
                                LoadB(instr.srcs[1], t);
                                enc.Remu(RV_T0, RV_T0, RV_T2);
                            }
                        }
                        StoreA(instr.dst, t);
                    }
                    break;
                }
                case LirOpcode::Pow: {
                    const TypeRef& t = instr.type;
                    bool isFloat = IsFloat(t);
                    if (isFloat) {
                        uint32_t sym = GetOrAddExtern("pow", RcuSymKind::ExternFunc);
                        EmitCallArgs(instr.srcs, instr.callConv);
                        uint32_t pcOff = enc.Size();
                        enc.Auipc(RV_RA, 0);
                        enc.Jalr(RV_RA, RV_RA, 0);
                        RelocPatch rp;
                        rp.patchOff = pcOff;
                        rp.symIdx = sym;
                        rp.relocType = 103;
                        relocPatches.push_back(rp);
                        StoreReturnValue(instr.dst, t);
                    }
                    else {
                        // Integer pow: inline exponentiation by squaring
                        LoadA(instr.srcs[0], t);  // base -> t0
                        LoadB(instr.srcs[1], t);  // exp -> t1
                        enc.Mv(RV_T2, RV_T0);     // save base in t2
                        enc.Li(RV_T0, 1);         // result = 1
                        // loop:
                        // beqz t1, done (28 bytes ahead = 28)
                        enc.Beq(RV_T1, RV_X0, 28);
                        // andi t3, t1, 1
                        enc.Andi(RV_T3, RV_T1, 1);
                        // beqz t3, square (8 bytes ahead = 8)
                        enc.Beq(RV_T3, RV_X0, 8);
                        // mul t0, t0, t2 (result *= base)
                        enc.Mul(RV_T0, RV_T0, RV_T2);
                        // mul t2, t2, t2 (base *= base)
                        enc.Mul(RV_T2, RV_T2, RV_T2);
                        // srai t1, t1, 1 (exp >>= 1)
                        enc.Srai(RV_T1, RV_T1, 1);
                        // j loop (-24 bytes = -24)
                        enc.Jal(RV_X0, -24);
                        // done:
                        StoreA(instr.dst, t);
                    }
                    break;
                }
                case LirOpcode::Shl:
                case LirOpcode::Shr: {
                    const TypeRef& t = instr.type;
                    LoadA(instr.srcs[0], t);
                    LdS0(RV_T1, Disp(instr.srcs[1]));
                    bool isShr = (instr.op == LirOpcode::Shr);
                    if (isShr && t.IsSigned()) {
                        enc.Sra(RV_T0, RV_T0, RV_T1);
                    }
                    else if (isShr) {
                        enc.Srl(RV_T0, RV_T0, RV_T1);
                    }
                    else {
                        enc.Sll(RV_T0, RV_T0, RV_T1);
                    }
                    StoreA(instr.dst, t);
                    break;
                }
                case LirOpcode::Neg: {
                    const TypeRef& t = instr.type;
                    if (IsFloat(t)) {
                        // Negate float by XOR with sign bit
                        // Use fsgnjn: fsgnjn.s ft0, ft0, ft0
                        if (t.kind == TypeRef::Kind::Float32) {
                            enc.Fsgnjn_s(RV_FT0, RV_FT0, RV_FT0);
                        }
                        else {
                            enc.Fsgnjn_d(RV_FT0, RV_FT0, RV_FT0);
                        }
                    }
                    else {
                        LoadA(instr.srcs[0], t);
                        enc.Neg(RV_T0, RV_T0);
                    }
                    StoreA(instr.dst, t);
                    break;
                }
                case LirOpcode::Not: {
                    LoadA(instr.srcs[0], instr.type);
                    enc.Seqz(RV_T0, RV_T0);
                    StoreA(instr.dst, TypeRef::MakeBool());
                    break;
                }
                case LirOpcode::BitNot: {
                    LoadA(instr.srcs[0], instr.type);
                    if (instr.type.IsBool()) {
                        enc.Xori(RV_T0, RV_T0, 1);
                    }
                    else {
                        enc.Not(RV_T0, RV_T0);
                    }
                    StoreA(instr.dst, instr.type);
                    break;
                }
                case LirOpcode::CmpEq:
                case LirOpcode::CmpNe:
                case LirOpcode::CmpLt:
                case LirOpcode::CmpLe:
                case LirOpcode::CmpGt:
                case LirOpcode::CmpGe: {
                    const TypeRef& lhsT = regTypes.contains(instr.srcs[0])
                        ? regTypes.at(instr.srcs[0]) : instr.type;
                    LoadA(instr.srcs[0], lhsT);
                    LoadB(instr.srcs[1], lhsT);
                    if (IsFloat(lhsT)) {
                        bool f32 = (lhsT.kind == TypeRef::Kind::Float32);
                        // Compare and set t0
                        switch (instr.op) {
                        case LirOpcode::CmpEq:
                            f32 ? enc.Feq_s(RV_T0, RV_FT0, 1)
                                : enc.Feq_d(RV_T0, RV_FT0, 1);
                            break;
                        case LirOpcode::CmpNe:
                            f32 ? enc.Feq_s(RV_T0, RV_FT0, 1)
                                : enc.Feq_d(RV_T0, RV_FT0, 1);
                            enc.Seqz(RV_T0, RV_T0); // invert
                            break;
                        case LirOpcode::CmpLt:
                            f32 ? enc.Flt_s(RV_T0, RV_FT0, 1)
                                : enc.Flt_d(RV_T0, RV_FT0, 1);
                            break;
                        case LirOpcode::CmpLe:
                            f32 ? enc.Fle_s(RV_T0, RV_FT0, 1)
                                : enc.Fle_d(RV_T0, RV_FT0, 1);
                            break;
                        case LirOpcode::CmpGt:
                            f32 ? enc.Flt_s(1, RV_FT0, 1)
                                : enc.Flt_d(1, RV_FT0, 1);
                            enc.Mv(RV_T0, RV_T1);
                            break;
                        case LirOpcode::CmpGe:
                            f32 ? enc.Fle_s(1, RV_FT0, 1)
                                : enc.Fle_d(1, RV_FT0, 1);
                            enc.Mv(RV_T0, RV_T1);
                            break;
                        }
                    }
                    else {
                        bool sig = lhsT.IsSigned();
                        switch (instr.op) {
                        case LirOpcode::CmpEq:
                            enc.Sub(RV_T0, RV_T0, RV_T1);
                            enc.Seqz(RV_T0, RV_T0);
                            break;
                        case LirOpcode::CmpNe:
                            enc.Sub(RV_T0, RV_T0, RV_T1);
                            enc.Snez(RV_T0, RV_T0);
                            break;
                        case LirOpcode::CmpLt:
                            if (sig) enc.Slt(RV_T0, RV_T0, RV_T1);
                            else enc.Sltu(RV_T0, RV_T0, RV_T1);
                            break;
                        case LirOpcode::CmpLe:
                            if (sig) enc.Slt(RV_T1, RV_T1, RV_T0); // swapped: b < a
                            else enc.Sltu(RV_T1, RV_T1, RV_T0);
                            enc.Seqz(RV_T0, RV_T1); // not (b < a)
                            break;
                        case LirOpcode::CmpGt:
                            if (sig) enc.Slt(RV_T0, RV_T1, RV_T0); // b < a
                            else enc.Sltu(RV_T0, RV_T1, RV_T0);
                            break;
                        case LirOpcode::CmpGe:
                            if (sig) enc.Slt(RV_T0, RV_T0, RV_T1); // a < b
                            else enc.Sltu(RV_T0, RV_T0, RV_T1);
                            enc.Seqz(RV_T0, RV_T0); // not (a < b)
                            break;
                        }
                    }
                    StoreA(instr.dst, TypeRef::MakeBool());
                    break;
                }
                case LirOpcode::Cast: {
                    const TypeRef& dstT = instr.type;
                    TypeRef srcT = regTypes.contains(instr.srcs[0])
                        ? regTypes.at(instr.srcs[0]) : dstT;
                    LoadA(instr.srcs[0], srcT);
                    bool srcFl = IsFloat(srcT), dstFl = IsFloat(dstT);
                    if (srcFl && !dstFl) {
                        if (srcT.kind == TypeRef::Kind::Float32)
                            enc.Fcvt_w_s(RV_T0, RV_FT0);
                        else
                            enc.Fcvt_w_d(RV_T0, RV_FT0);
                    }
                    else if (!srcFl && dstFl) {
                        if (dstT.kind == TypeRef::Kind::Float32)
                            enc.Fcvt_l_s(RV_FT0, RV_T0);
                        else
                            enc.Fcvt_l_d(RV_FT0, RV_T0);
                    }
                    else if (srcFl && dstFl) {
                        if (srcT.kind == TypeRef::Kind::Float32 &&
                            dstT.kind == TypeRef::Kind::Float64)
                            enc.Fcvt_d_s(RV_FT0, RV_FT0);
                        else if (srcT.kind == TypeRef::Kind::Float64 &&
                                 dstT.kind == TypeRef::Kind::Float32)
                            enc.Fcvt_s_d(RV_FT0, RV_FT0);
                    }
                    StoreA(instr.dst, dstT);
                    break;
                }
                case LirOpcode::Call: {
                    // Builtins
                    if (instr.strArg == "FloatBits64" && instr.srcs.size() == 1) {
                        FldS0(RV_FT0, Disp(instr.srcs[0]));
                        enc.Fmv_x_d(RV_T0, RV_FT0);
                        if (instr.dst != LirNoReg && !instr.type.IsOpaque()) {
                            StoreReturnValue(instr.dst, instr.type);
                        }
                        break;
                    }
                    if (instr.strArg == "FloatFromBits64" && instr.srcs.size() == 1) {
                        LdS0(RV_T0, Disp(instr.srcs[0]));
                        enc.Fmv_d_x(RV_FT0, RV_T0);
                        if (instr.dst != LirNoReg && !instr.type.IsOpaque()) {
                            StoreReturnValue(instr.dst, instr.type);
                        }
                        break;
                    }
                    if (instr.strArg == "FloatBits32" && instr.srcs.size() == 1) {
                        FlwS0(RV_FT0, Disp(instr.srcs[0]));
                        enc.Fmv_x_w(RV_T0, RV_FT0);
                        // Zero-extend to 64-bit (fmv.x.w sign-extends on RV64)
                        // Actually on RV64, fmv.x.w sign-extends the 32-bit value.
                        // We want zero-extended.
                        // slli + srli to zero-extend
                        enc.Slli(RV_T0, RV_T0, 32);
                        enc.Srli(RV_T0, RV_T0, 32);
                        if (instr.dst != LirNoReg && !instr.type.IsOpaque()) {
                            StoreReturnValue(instr.dst, instr.type);
                        }
                        break;
                    }
                    if (instr.strArg == "FloatFromBits32" && instr.srcs.size() == 1) {
                        LwS0(RV_T0, Disp(instr.srcs[0]));
                        enc.Fmv_d_x(RV_FT0, RV_T0);
                        if (instr.dst != LirNoReg && !instr.type.IsOpaque()) {
                            StoreReturnValue(instr.dst, instr.type);
                        }
                        break;
                    }
                    // Regular function call
                    const bool hiddenReturn =
                        instr.dst != LirNoReg &&
                        SizeOfRuntime(instr.type) == 16;
                    if (hiddenReturn) {
                        // Pass hidden pointer in a0
                        AddiS0(RV_A0, Disp(instr.dst));
                        EmitCallArgs(instr.srcs, instr.callConv, 1);
                    }
                    else {
                        EmitCallArgs(instr.srcs, instr.callConv);
                    }
                    uint32_t symIdx;
                    if (auto it = funcSyms.find(instr.strArg);
                        it != funcSyms.end()) {
                        symIdx = it->second;
                    }
                    else {
                        symIdx = GetOrAddExtern(instr.strArg, RcuSymKind::ExternFunc);
                    }

                    uint32_t pcOff = enc.Size();
                    enc.Auipc(RV_RA, 0);
                    enc.Jalr(RV_RA, RV_RA, 0);
                    RelocPatch rp;
                    rp.patchOff = pcOff;
                    rp.symIdx = symIdx;
                    rp.relocType = 103; // AUIPC+JALR call
                    relocPatches.push_back(rp);
                    if (instr.dst != LirNoReg && !instr.type.IsOpaque() &&
                        !hiddenReturn) {
                        StoreReturnValue(instr.dst, instr.type);
                    }
                    break;
                }
                case LirOpcode::CallIndirect: {
                    if (instr.srcs.empty()) break;
                    LirReg callee = instr.srcs[0];
                    std::vector<LirReg> args(instr.srcs.begin() + 1,
                                             instr.srcs.end());
                    const bool hiddenReturn =
                        instr.dst != LirNoReg &&
                        SizeOfRuntime(instr.type) == 16;
                    if (hiddenReturn) {
                        AddiS0(RV_A0, Disp(instr.dst));
                        EmitCallArgs(args, instr.callConv, 1);
                    }
                    else {
                        EmitCallArgs(args, instr.callConv);
                    }
                    LdS0(RV_T2, Disp(callee));
                    enc.Jalr(RV_RA, RV_T2, 0);
                    if (instr.dst != LirNoReg && !instr.type.IsOpaque() &&
                        !hiddenReturn) {
                        StoreReturnValue(instr.dst, instr.type);
                    }
                    break;
                }
                case LirOpcode::GlobalAddr: {
                    uint32_t symIdx;
                    if (auto dataIt = dataSyms.find(instr.strArg);
                        dataIt != dataSyms.end()) {
                        symIdx = dataIt->second;
                    }
                    else if (auto funcIt = funcSyms.find(instr.strArg);
                             funcIt != funcSyms.end()) {
                        symIdx = funcIt->second;
                    }
                    else {
                        symIdx = GetOrAddExtern(instr.strArg, RcuSymKind::ExternData);
                    }
                    uint32_t pcOff = enc.Size();
                    enc.Auipc(RV_T0, 0);
                    enc.Addi(RV_T0, RV_T0, 0);
                    RelocPatch rp;
                    rp.patchOff = pcOff;
                    rp.symIdx = symIdx;
                    rp.relocType = 100; // AUIPC+ADDI pair
                    relocPatches.push_back(rp);
                    SdS0(RV_T0, Disp(instr.dst));
                    break;
                }
                case LirOpcode::FieldPtr: {
                    LirReg base = instr.srcs[0];
                    auto typeIt = regTypes.find(base);
                    if (typeIt != regTypes.end() &&
                        typeIt->second.kind == TypeRef::Kind::Pointer) {
                        LdS0(RV_T0, Disp(base));
                    } else {
                        AddiS0(RV_T0, Disp(base));
                    }
                    int off = FieldOffset(base, instr.strArg);
                    if (off != 0) {
                        enc.Addi(RV_T0, RV_T0, off);
                    }
                    SdS0(RV_T0, Disp(instr.dst));
                    break;
                }
                case LirOpcode::IndexPtr: {
                    LirReg base = instr.srcs[0];
                    LirReg idx = instr.srcs[1];
                    int elemSz = (instr.type.kind == TypeRef::Kind::Pointer &&
                                  !instr.type.inner.empty())
                        ? SizeOfRuntime(instr.type.inner[0]) : 8;
                    if (elemSz < 1) elemSz = 1;
                    LdS0(RV_T0, Disp(base));
                    LdS0(RV_T1, Disp(idx));
                    enc.Li(RV_T2, elemSz);
                    enc.Mul(RV_T1, RV_T1, RV_T2);
                    enc.Add(RV_T0, RV_T0, RV_T1);
                    SdS0(RV_T0, Disp(instr.dst));
                    break;
                }
                case LirOpcode::Phi:
                    break;
                default:
                    break;
                }
            }

            void GenTerm(uint32_t blockIdx,
                         const LirTerminator& term,
                         const LirFunc& func) {
                (void)func;
                switch (term.kind) {
                case LirTermKind::Jump: {
                    EmitPhiMoves(blockIdx, term.trueTarget);
                    uint32_t patchOff = enc.Size();
                    enc.Jal(RV_X0, 0); // placeholder
                    jumpPatches.push_back({patchOff, term.trueTarget, 0, 0, 0, 0});
                    break;
                }
                case LirTermKind::Branch: {
                    // Load condition
                    const TypeRef condT = regTypes.contains(term.cond)
                        ? regTypes.at(term.cond) : TypeRef::MakeBool();
                    int condSz = SizeOf(condT);
                    int32_t d = Disp(term.cond);
                    uint8_t base;
                    int32_t adj;
                    PrepSOffset(d, base, adj);
                    if (condSz <= 1) {
                        enc.Lbu(RV_T0, base, adj);
                    }
                    else if (condSz == 2) {
                        enc.Lhu(RV_T0, base, adj);
                    }
                    else if (condSz == 4) {
                        enc.Lwu(RV_T0, base, adj);
                    }
                    else {
                        enc.Ld(RV_T0, base, adj);
                    }
                    bool truePhi = HasPhiMoves(blockIdx, term.trueTarget);
                    bool falsePhi = HasPhiMoves(blockIdx, term.falseTarget);
                    if (!truePhi && !falsePhi) {
                        // beqz t0, falseTarget
                        uint32_t bOff = enc.Size();
                        enc.Beq(RV_T0, RV_X0, 0);
                        jumpPatches.push_back(
                            {bOff, term.falseTarget, 1, RV_T0, RV_X0, F3_BEQ});
                        // j trueTarget
                        uint32_t jOff = enc.Size();
                        enc.Jal(RV_X0, 0);
                        jumpPatches.push_back(
                            {jOff, term.trueTarget, 0, 0, 0, 0});
                    }
                    else {
                        uint32_t bOff = enc.Size();
                        enc.Beq(RV_T0, RV_X0, 0);
                        jumpPatches.push_back(
                            {bOff, term.falseTarget, 1, RV_T0, RV_X0, F3_BEQ});
                        // true trampoline
                        EmitPhiMoves(blockIdx, term.trueTarget);
                        uint32_t jOff = enc.Size();
                        enc.Jal(RV_X0, 0);
                        jumpPatches.push_back(
                            {jOff, term.trueTarget, 0, 0, 0, 0});
                    }
                    break;
                }
                case LirTermKind::Return: {
                    if (term.retVal && *term.retVal != LirNoReg) {
                        if (hiddenReturnOff != 0 &&
                            SizeOfRuntime(term.retType) == 16) {
                            StoreHiddenReturnValue(*term.retVal, term.retType);
                        }
                        else {
                            LoadReturnValue(*term.retVal, term.retType);
                        }
                    }
                    // Epilogue
                    if (frameSize + 16 <= 2047) {
                        enc.Addi(RV_SP, RV_SP, frameSize + 16);
                        enc.Ld(RV_RA, RV_SP, -8);
                        enc.Ld(RV_S0, RV_SP, -16);
                    }
                    else {
                        // s0 = initial_sp, restore sp from it
                        enc.Mv(RV_SP, RV_S0);
                        enc.Ld(RV_RA, RV_SP, -8);
                        enc.Ld(RV_S0, RV_SP, -16);
                    }
                    enc.Ret();
                    break;
                }
                case LirTermKind::Switch: {
                    LdS0(RV_T0, Disp(term.cond));
                    for (const auto& c : term.cases) {
                        const std::uint64_t bits =
                            ParseIntegerLiteralBits(c.value).value_or(0);
                        enc.Li(RV_T1, static_cast<int64_t>(bits));
                        uint32_t bOff = enc.Size();
                        enc.Bne(RV_T0, RV_T1, 0);
                        jumpPatches.push_back(
                            {bOff, c.target, 1, RV_T0, RV_T1, F3_BNE});
                    }
                    EmitPhiMoves(blockIdx, term.defaultTarget);
                    uint32_t jOff = enc.Size();
                    enc.Jal(RV_X0, 0);
                    jumpPatches.push_back(
                        {jOff, term.defaultTarget, 0, 0, 0, 0});
                    break;
                }
                }
            }

            void GenFunc(const LirFunc& func) {
                if (func.isExtern) {
                    GetOrAddExtern(func.name, RcuSymKind::ExternFunc, func.dll);
                    return;
                }
                PrepassFunc(func);
                if (func.name.find("Main") != std::string::npos ||
                    func.name.find("Copy") != std::string::npos ||
                    func.name.find("+__String") != std::string::npos ||
                    func.name.find("ToString") != std::string::npos ||
                    func.name.find("Print__uint64") != std::string::npos ||
                    func.name.find("String::From") != std::string::npos ||
                    func.name.find("Print__String") != std::string::npos ||
                    func.name.find("Print___char8_uint") != std::string::npos) {
                    fprintf(stderr, "=== %s layout ===\n  frameSize=%d hiddenReturnOff=%d\n",
                            func.name.c_str(), frameSize, hiddenReturnOff);
                    for (auto& e : slotMap)
                        fprintf(stderr, "  slot r%d off=%d\n", e.first, e.second);
                    for (auto& e : allocaData)
                        fprintf(stderr, "  alloca r%d dataOff=%d\n", e.first, e.second);
                    for (uint32_t bi = 0; bi < func.blocks.size(); ++bi) {
                        fprintf(stderr, "  block %d:\n", bi);
                        for (const auto& instr : func.blocks[bi].instrs) {
                            fprintf(stderr, "    op=%d", static_cast<int>(instr.op));
                            if (instr.dst != LirNoReg)
                                fprintf(stderr, " r%d", instr.dst);
                            if (!instr.srcs.empty()) {
                                fprintf(stderr, " [");
                                for (size_t si = 0; si < instr.srcs.size(); ++si) {
                                    if (si > 0) fprintf(stderr, ",");
                                    fprintf(stderr, "r%d", instr.srcs[si]);
                                }
                                fprintf(stderr, "]");
                            }
                            fprintf(stderr, " %s", instr.type.ToString().c_str());
                            if (!instr.strArg.empty())
                                fprintf(stderr, " \"%s\"", instr.strArg.c_str());
                            fprintf(stderr, "\n");
                        }
                        if (func.blocks[bi].term.has_value()) {
                            fprintf(stderr, "    term: kind=%d target=%d\n",
                                    func.blocks[bi].term->kind,
                                    func.blocks[bi].term->trueTarget);
                        }
                    }
                }
                jumpPatches.clear();
                relocPatches.clear();
                uint32_t funcStart = enc.Size();

                RcuSymbol sym;
                sym.name = func.name;
                sym.sectionIdx = RCU_TEXT_IDX;
                sym.value = funcStart;
                sym.kind = RcuSymKind::Func;
                sym.visibility =
                    func.isPublic ? RcuSymVis::Global : RcuSymVis::Local;
                sym.typeName = func.returnType.ToString();
                if (auto it = funcSyms.find(func.name); it != funcSyms.end()) {
                    symbols[it->second] = std::move(sym);
                }
                else {
                    funcSyms[func.name] = AddSymbol(std::move(sym));
                }

                // Prologue
                // Total frame = frameSize + 16 (for ra and s0)
                int32_t totalFrame = frameSize + 16;
                // Align to 16 bytes
                totalFrame = AlignUp(totalFrame, 16);
                if (totalFrame < 16) totalFrame = 16;

                if (totalFrame <= 2047) {
                    enc.Addi(RV_SP, RV_SP, -totalFrame);
                    enc.Sd(RV_RA, RV_SP, totalFrame - 8);
                    enc.Sd(RV_S0, RV_SP, totalFrame - 16);
                    enc.Addi(RV_S0, RV_SP, totalFrame);
                }
                else {
                    enc.Li(RV_T0, totalFrame);
                    enc.Sub(RV_SP, RV_SP, RV_T0);
                    // Save old s0 at initial_sp - 16
                    enc.Addi(RV_T1, RV_T0, -16);
                    enc.Add(RV_T1, RV_SP, RV_T1);
                    enc.Sd(RV_S0, RV_T1, 0);
                    // Save ra at initial_sp - 8
                    enc.Addi(RV_T1, RV_T0, -8);
                    enc.Add(RV_T1, RV_SP, RV_T1);
                    enc.Sd(RV_RA, RV_T1, 0);
                    // Set s0 = initial_sp
                    enc.Add(RV_S0, RV_SP, RV_T0);
                }

                // Spill arguments from a0-a7 / fa0-fa7 to stack
                // Use block-local state from PrepassFunc: the params
                // RISC-V: 16-byte return uses hidden pointer in a0;
                // parameters start from a1.
                int intIdx = (hiddenReturnOff != 0) ? 1 : 0, fltIdx = 0;
                if (hiddenReturnOff != 0) {
                    int32_t hOff = -hiddenReturnOff;
                    uint8_t hBase;
                    int32_t hAdj;
                    PrepSOffset(hOff, hBase, hAdj);
                    enc.Sd(RV_A0, hBase, hAdj);
                }
                for (const auto& p : func.params) {
                    int sz = SizeOfRuntime(p.type);
                    // Store parameter value to the alloca DATA REGION
                    auto adIt = allocaData.find(p.reg);
                    int32_t d = (adIt != allocaData.end())
                        ? -(int32_t)adIt->second - 16
                        : Disp(p.reg);
                    if (func.name.find("String") != std::string::npos)
                        fprintf(stderr, "  PROLOGUE func=%s param r%d type=%s sz=%d dataOff=%d disp=%d\n",
                                func.name.c_str(), p.reg, p.type.ToString().c_str(), sz,
                                (int)(adIt != allocaData.end() ? adIt->second : -1), d);
                    uint8_t base;
                    int32_t adj;
                    PrepSOffset(d, base, adj);
                    if (IsFloat(p.type)) {
                        if (fltIdx < 8) {
                            if (sz == 4) {
                                enc.Fsw(RV_FA0 + fltIdx, base, adj);
                            }
                            else {
                                enc.Fsd(RV_FA0 + fltIdx, base, adj);
                            }
                            ++fltIdx;
                        }
                    }
                    else {
                        if (intIdx < 8) {
                            if (sz == 16) {
                                // LIR lowering passes alloca pointers (8 bytes)
                                // for struct value params like String. The
                                // register (A0+intIdx) holds a pointer to the
                                // data region. Dereference it and store the
                                // 16-byte struct value.
                                // Debug: print a2, [a2], [a2+8] via write(2,...)
                                enc.Mv(RV_T2, RV_A0 + intIdx);
                                // store a2 to a known slot for debug
                                enc.Sd(RV_T2, RV_SP, 0);
                                enc.Ld(RV_T0, RV_T2, 0);
                                enc.Sd(RV_T0, base, adj);
                                enc.Ld(RV_T0, RV_T2, 8);
                                enc.Sd(RV_T0, base, adj + 8);
                                ++intIdx;
                            }
                            else if (sz == 4) {
                                enc.Sw(RV_A0 + intIdx, base, adj);
                                ++intIdx;
                            }
                            else if (sz == 2) {
                                enc.Sh(RV_A0 + intIdx, base, adj);
                                ++intIdx;
                            }
                            else if (sz == 1) {
                                enc.Sb(RV_A0 + intIdx, base, adj);
                                ++intIdx;
                            }
                            else {
                                enc.Sd(RV_A0 + intIdx, base, adj);
                                ++intIdx;
                            }
                        }
                    }
                }

                // Basic blocks
                blockOffsets.assign(func.blocks.size(), 0);
                for (uint32_t bi = 0; bi < func.blocks.size(); ++bi) {
                    blockOffsets[bi] = enc.Size();
                    const auto& block = func.blocks[bi];
                    for (const auto& instr : block.instrs) {
                        GenInstr(instr);
                    }
                    if (block.term) {
                        GenTerm(bi, *block.term, func);
                    }
                }
                PatchJumps();
                // Flush reloc patches to textRelocs (one func at a time)
                for (auto& rp : relocPatches) {
                    textRelocs.push_back({rp.patchOff, rp.symIdx,
                                          static_cast<uint16_t>(rp.relocType), 0});
                }
                relocPatches.clear();

                // Update symbol size
                for (auto& s : symbols) {
                    if (s.name == func.name && s.sectionIdx == RCU_TEXT_IDX &&
                        s.value == funcStart) {
                        s.size = enc.Size() - funcStart;
                        break;
                    }
                }
            }

            void EmitVtables() {
                for (const auto& vt : mod.vtables) {
                    AlignRodata(8);
                    RcuSymbol sym;
                    sym.name = vt.label;
                    sym.sectionIdx = RCU_RODATA_IDX;
                    sym.value = static_cast<uint32_t>(rodataData.size());
                    sym.size = static_cast<uint32_t>(vt.methods.size() * 8);
                    sym.kind = RcuSymKind::Const;
                    sym.visibility = RcuSymVis::Global;
                    const uint32_t vtSym = AddSymbol(std::move(sym));
                    dataSyms[vt.label] = vtSym;

                    for (const auto& method : vt.methods) {
                        const uint32_t slotOff =
                            static_cast<uint32_t>(rodataData.size());
                        for (int i = 0; i < 8; ++i) rodataData.push_back(0);
                        uint32_t methodSym;
                        if (auto it = funcSyms.find(method);
                            it != funcSyms.end()) {
                            methodSym = it->second;
                        }
                        else {
                            methodSym =
                                GetOrAddExtern(method, RcuSymKind::ExternFunc);
                        }
                        AddRodataReloc(slotOff, methodSym, RcuRelType::Abs64);
                    }
                }
            }

            void GenModule() {
                BuildLayouts();
                PredeclareFunctions();
                for (const auto& ev : mod.externVars) {
                    GetOrAddExtern(ev.name, RcuSymKind::ExternData);
                }
                for (const auto& c : mod.consts) {
                    RcuSymbol s;
                    s.name = c.name;
                    s.sectionIdx = RCU_DATA_IDX;
                    s.value = static_cast<uint32_t>(dataData.size());
                    s.kind = RcuSymKind::Const;
                    s.visibility =
                        c.isPublic ? RcuSymVis::Global : RcuSymVis::Local;
                    s.typeName = c.type.ToString();
                    for (int i = 0; i < 8; ++i) dataData.push_back(0);
                    s.size = 8;
                    AddSymbol(std::move(s));
                }
                EmitVtables();
                for (const auto& func : mod.funcs) {
                    GenFunc(func);
                }
            }

            void PatchRelocations(const RcuFile& file) {
                // Apply relocPatches to textRelocs
                // For each patch, encode the reloc type and symbol
                for (auto& rp : relocPatches) {
                    textRelocs.push_back({rp.patchOff, rp.symIdx,
                                          static_cast<uint16_t>(rp.relocType), 0});
                }
            }
        };

        RcuFile RiscVCodeGen::Generate() {
            GenModule();

            RcuFile file;
            file.arch = RcuArch::RISCV64;
            file.sourcePath = mod.name;
            file.packageName = pkgName;
            file.buildTimestamp = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count());
            {
                std::string ver = RUX_VERSION;
                unsigned M = 0, mi = 0, p = 0;
                auto parseNum = [](const char* s, unsigned& out) -> const char* {
                    while (*s && (*s < '0' || *s > '9')) ++s;
                    while (*s >= '0' && *s <= '9') {
                        out = out * 10 + static_cast<unsigned>(*s - '0');
                        ++s;
                    }
                    return s;
                };
                const char* c1 = parseNum(ver.c_str(), M);
                const char* c2 = parseNum(c1, mi);
                parseNum(c2, p);
                file.ruxVersion = (M << 16) | (mi << 8) | p;
            }

            // Apply pending reloc patches
            PatchRelocations(file);

            // Build sections
            {
                RcuSection text;
                text.name = ".text";
                text.type = RcuSecType::Text;
                text.flags = RcuSecFlag::Alloc | RcuSecFlag::Exec | RcuSecFlag::Read;
                text.alignment = 16;
                text.data = std::move(textData);
                text.relocs = std::move(textRelocs);
                file.sections.push_back(std::move(text));
            }
            {
                RcuSection rodata;
                rodata.name = ".rodata";
                rodata.type = RcuSecType::RoData;
                rodata.flags = RcuSecFlag::Alloc | RcuSecFlag::Read;
                rodata.alignment = 8;
                rodata.data = std::move(rodataData);
                rodata.relocs = std::move(rodataRelocs);
                file.sections.push_back(std::move(rodata));
            }
            {
                RcuSection data;
                data.name = ".data";
                data.type = RcuSecType::Data;
                data.flags = RcuSecFlag::Alloc | RcuSecFlag::Read | RcuSecFlag::Write;
                data.alignment = 8;
                data.data = std::move(dataData);
                file.sections.push_back(std::move(data));
            }

            file.symbols = std::move(symbols);
            return file;
        }

    } // namespace

    RcuFile GenerateRiscV(const LirModule& mod,
                          const std::vector<LirStructDecl>& structDecls,
                          const std::vector<std::string>& interfaceNames,
                          const std::string& pkgName) {
        RiscVCodeGen gen(mod, structDecls, interfaceNames, pkgName);
        return gen.Generate();
    }

} // namespace Rux
