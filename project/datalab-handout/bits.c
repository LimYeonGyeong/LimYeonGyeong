/* * CS:APP Data Lab 
 * * <Please put your name and userid here>
 * * bits.c - Source file with your solutions to the Lab.
 * This is the file you will hand in to your instructor.
 *
 * WARNING: Do not include the <stdio.h> header; it confuses the dlc
 * compiler. You can still use printf for debugging without including
 * <stdio.h>, although you might get a compiler warning. In general,
 * it's not good practice to ignore compiler warnings, but in this
 * case it's OK.  
 */

#if 0
/*
 * Instructions to Students:
 *
 * STEP 1: Read the following instructions carefully.
 */

#endif

/* * bitNor - ~(x|y) using only ~ and & 
 * Example: bitNor(0x6, 0x5) = 0xFFFFFFF8
 * Legal ops: ~ &
 * Max ops: 8
 * Rating: 1
 */
int bitNor(int x, int y) {
  /* De Morgan's Law: ~(x|y) = ~x & ~y */
  return (~x & ~y);
}

/* * bitAnd - x&y using only ~ and | 
 * Example: bitAnd(6, 5) = 4
 * Legal ops: ~ |
 * Max ops: 8
 * Rating: 1
 */
int bitAnd(int x, int y) {
  /* De Morgan's Law: x&y = ~(~x | ~y) */
  return ~(~x | ~y);
}

/* * thirdBits - return word with every third bit (starting from the LSB) set to 1
 * Legal ops: ! ~ & ^ | + << >>
 * Max ops: 8
 * Rating: 1
 */
int thirdBits(void) {
  /* Construct 0x49249249 using 8-bit constants and shifts */
  int x = 0x49;
  x = (x << 9) | 0x49;
  x = (x << 18) | (x >> 9);
  return x;
}

/* * bitMatch - Create mask indicating which bits in x match those in y
 * using only ~ and & 
 * Example: bitMatch(0x7, 0xE) = 0x6
 * Legal ops: ~ & |
 * Max ops: 14
 * Rating: 1
 */
int bitMatch(int x, int y) {
  /* (x & y) identifies matches of 1s, (~x & ~y) identifies matches of 0s */
  return (x & y) | (~x & ~y);
}

/* * allOddBits - return 1 if all odd-numbered bits in word set to 1
 * where bits are numbered from 0 (least significant) to 31 (most significant)
 * Examples allOddBits(0xFFFFFFFD) = 0, allOddBits(0xAAAAAAAA) = 1
 * Legal ops: ! ~ & ^ | + << >>
 * Max ops: 12
 * Rating: 2
 */
int allOddBits(int x) {
  /* Create 0xAAAAAAAA mask and compare with x */
  int mask = 0xAA;
  mask = (mask << 8) | 0xAA;
  mask = (mask << 16) | mask;
  return !((x & mask) ^ mask);
}

/* * dividePower2 - Compute x/(2^n), for 0 <= n <= 30
 * Round toward zero
 * Examples: dividePower2(15,1) = 7, dividePower2(-33,4) = -2
 * Legal ops: ! ~ & ^ | + << >>
 * Max ops: 15
 * Rating: 2
 */
int dividePower2(int x, int n) {
  /* Add bias (2^n - 1) if x is negative to round toward zero */
  int bias = (x >> 31) & ((1 << n) + ~0);
  return (x + bias) >> n;
}

/* * isNegative - return 1 if x < 0, return 0 otherwise 
 * Example: isNegative(-1) = 1.
 * Legal ops: ! ~ & ^ | + << >>
 * Max ops: 6
 * Rating: 2
 */
int isNegative(int x) {
  /* Check the sign bit (MSB) */
  return (x >> 31) & 1;
}

/* * addOK - Determine if can compute x+y without overflow
 * Example: addOK(0x80000000,0x80000000) = 0,
 * addOK(0x80000000,0x70000000) = 1, 
 * Legal ops: ! ~ & ^ | + << >>
 * Max ops: 20
 * Rating: 3
 */
int addOK(int x, int y) {
  /* Overflow occurs if signs of x and y are same, but sum sign is different */
  int sum = x + y;
  int x_sign = x >> 31;
  int y_sign = y >> 31;
  int s_sign = sum >> 31;
  return !!(x_sign ^ y_sign) | !(x_sign ^ s_sign);
}

/* * isLess - if x < y  then return 1, else return 0 
 * Example: isLess(4,5) = 1.
 * Legal ops: ! ~ & ^ | + << >>
 * Max ops: 24
 * Rating: 3
 */
int isLess(int x, int y) {
  /* Handle different signs and same signs separately to avoid overflow */
  int x_sign = x >> 31;
  int y_sign = y >> 31;
  int same_sign = !(x_sign ^ y_sign);
  return ((!same_sign) & x_sign) | (same_sign & ((x + ~y + 1) >> 31));
}

/* * absVal - absolute value of x
 * Example: absVal(-1) = 1.
 * You may assume -TMax <= x <= TMax
 * Legal ops: ! ~ & ^ | + << >>
 * Max ops: 10
 * Rating: 4
 */
int absVal(int x) {
  /* Using mask (all 1s for negative, all 0s for positive) to flip and add 1 */
  int mask = x >> 31;
  return (x ^ mask) + (~mask + 1);
}