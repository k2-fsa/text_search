// Copied and modified from
// https://github.com/Martinsos/edlib/blob/master/test/SimpleEditDistance.h and
// https://github.com/Martinsos/edlib/blob/master/test/runTests.cpp

// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <gtest/gtest.h>

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#include "textsearch/csrc/edlib.h"

using namespace std;
using namespace edlib;

static int max(int a, int b) { return a > b ? a : b; }

static int min(int x, int y) { return x < y ? x : y; }

static int min3(int x, int y, int z) { return min(x, min(y, z)); }

template <class Element = char>
static int calcEditDistanceSimple(const Element *query, int queryLength,
                                  const Element *target, int targetLength,
                                  const edlib::EdlibAlignMode mode, int *score,
                                  int **positions_, int *numPositions_) {
  int bestScore = -1;
  std::vector<int> positions;

  // Handle as a special situation when one of the sequences has length 0.
  if (queryLength == 0 || targetLength == 0) {
    if (mode == edlib::EDLIB_MODE_NW) {
      *score = std::max(queryLength, targetLength);
      *positions_ = new int[1];
      *positions_[0] = targetLength - 1;
      *numPositions_ = 1;
    } else if (mode == edlib::EDLIB_MODE_SHW || mode == edlib::EDLIB_MODE_HW) {
      *score = queryLength;
      *positions_ = new int[1];
      *positions_[0] = -1;
      *numPositions_ = 1;
    } else {
      return EDLIB_STATUS_ERROR;
    }
    return EDLIB_STATUS_OK;
  }

  int *C = new int[queryLength];
  int *newC = new int[queryLength];

  // set first column (column zero)
  for (int i = 0; i < queryLength; i++) {
    C[i] = i + 1;
  }

  for (int c = 0; c < targetLength; c++) { // for each column
    newC[0] = min3((mode == edlib::EDLIB_MODE_HW ? 0 : c + 1) + 1, // up
                   (mode == edlib::EDLIB_MODE_HW ? 0 : c) +
                       (target[c] == query[0] ? 0 : 1), // up left
                   C[0] + 1);                           // left
    for (int r = 1; r < queryLength; r++) {
      newC[r] = min3(newC[r - 1] + 1,                            // up
                     C[r - 1] + (target[c] == query[r] ? 0 : 1), // up left
                     C[r] + 1);                                  // left
    }

    if (mode != edlib::EDLIB_MODE_NW ||
        c == targetLength - 1) { // For NW check only last column
      int newScore = newC[queryLength - 1];
      if (bestScore == -1 || newScore <= bestScore) {
        if (newScore < bestScore) {
          positions.clear();
        }
        bestScore = newScore;
        positions.push_back(c);
      }
    }

    int *tmp = C;
    C = newC;
    newC = tmp;
  }

  delete[] C;
  delete[] newC;

  *score = bestScore;
  if (positions.size() > 0) {
    *positions_ = new int[positions.size()];
    *numPositions_ = static_cast<int>(positions.size());
    copy(positions.begin(), positions.end(), *positions_);
  } else {
    *positions_ = NULL;
    *numPositions_ = 0;
  }

  return EDLIB_STATUS_OK;
}

/**
 * Checks if alignment is correct.
 */
template <class Element>
static bool checkAlignment(const Element *query, int queryLength,
                           const Element *target, int score, int pos,
                           EdlibAlignMode mode, unsigned char *alignment,
                           int alignmentLength) {
  int alignScore = 0;
  int qIdx = queryLength - 1;
  int tIdx = pos;
  for (int i = alignmentLength - 1; i >= 0; i--) {
    if (alignment[i] == EDLIB_EDOP_MATCH) { // match
      if (query[qIdx] != target[tIdx]) {
        printf("Should be match but is a mismatch! (tIdx, qIdx, i): (%d, %d, "
               "%d)\n",
               tIdx, qIdx, i);
        return false;
      }
      qIdx--;
      tIdx--;
    } else if (alignment[i] == EDLIB_EDOP_MISMATCH) { // mismatch
      if (query[qIdx] == target[tIdx]) {
        printf("Should be mismatch but is a match! (tIdx, qIdx, i): (%d, %d, "
               "%d)\n",
               tIdx, qIdx, i);
        return false;
      }
      alignScore += 1;
      qIdx--;
      tIdx--;
    } else if (alignment[i] == EDLIB_EDOP_INSERT) {
      alignScore += 1;
      qIdx--;
    } else if (alignment[i] == EDLIB_EDOP_DELETE) {
      if (!(mode == EDLIB_MODE_HW && qIdx == -1))
        alignScore += 1;
      tIdx--;
    }
    if (tIdx < -1 || qIdx < -1) {
      printf(
          "Alignment went outside of matrix! (tIdx, qIdx, i): (%d, %d, %d)\n",
          tIdx, qIdx, i);
      return false;
    }
  }
  if (qIdx != -1) {
    printf("Alignment did not reach end!\n");
    return false;
  }
  if (alignScore != score) {
    printf("Wrong score in alignment! %d should be %d\n", alignScore, score);
    return false;
  }
  if (alignmentLength > 0 && alignment[0] == EDLIB_EDOP_INSERT && tIdx >= 0) {
    printf("Alignment starts with insertion in target, while it could start "
           "with mismatch!\n");
    return false;
  }
  return true;
}

/**
 * @param alignment
 * @param alignmentLength
 * @param endLocation
 * @return Return start location of alignment in target, if there is none return
 * -1.
 */
static int getAlignmentStart(const unsigned char *alignment,
                             int alignmentLength, int endLocation) {
  int startLocation = endLocation + 1;
  for (int i = 0; i < alignmentLength; i++) {
    if (alignment[i] != EDLIB_EDOP_INSERT) {
      startLocation--;
    }
  }
  if (startLocation > endLocation) {
    return -1;
  }
  return startLocation;
}

static bool executeTest(const char *query, int queryLength, const char *target,
                        int targetLength, EdlibAlignMode mode) {
  bool pass = true;

  int scoreSimple = -1;
  int numLocationsSimple = 0;
  int *endLocationsSimple = NULL;
  calcEditDistanceSimple(query, queryLength, target, targetLength, mode,
                         &scoreSimple, &endLocationsSimple,
                         &numLocationsSimple);

  EdlibAlignResult result =
      edlibAlign(query, queryLength, target, targetLength,
                 edlibNewAlignConfig<char>(-1, mode, EDLIB_TASK_PATH, NULL, 0));

  if (result.editDistance != scoreSimple) {
    pass = false;
    printf("Scores: expected %d, got %d\n", scoreSimple, result.editDistance);
  } else if (result.numLocations != numLocationsSimple) {
    pass = false;
    printf("Number of locations: expected %d, got %d\n", numLocationsSimple,
           result.numLocations);
  } else {
    for (int i = 0; i < result.numLocations; i++) {
      if (result.endLocations[i] != endLocationsSimple[i]) {
        pass = false;
        printf("End locations at %d are not equal! Expected %d, got %d\n", i,
               endLocationsSimple[i], result.endLocations[1]);
        break;
      }
    }
  }
  if (result.alignment) {
    if (!checkAlignment(query, queryLength, target, result.editDistance,
                        result.endLocations[0], mode, result.alignment,
                        result.alignmentLength)) {
      pass = false;
      printf("Alignment is not correct\n");
    }
    int alignmentStart = getAlignmentStart(
        result.alignment, result.alignmentLength, result.endLocations[0]);
    if (result.startLocations[0] != alignmentStart) {
      pass = false;
      printf(
          "Start location (%d) is not consistent with alignment start (%d)\n",
          result.startLocations[0], alignmentStart);
    }
  }

  delete[] endLocationsSimple;
  edlibFreeAlignResult(result);
  return pass;
}

template <class Element, class AlphabetIdx>
static void fillRandomly(Element *seq, int seqLength,
                         AlphabetIdx alphabetSize) {
  for (int i = 0; i < seqLength; i++)
    seq[i] = static_cast<Element>(rand() % alphabetSize);
}

// Returns true if all tests passed, false otherwise.
template <class Element, class AlphabetIdx>
bool runRandomTests(int numTests, EdlibAlignMode mode, bool findAlignment,
                    AlphabetIdx alphabetSize) {
  int numTestsFailed = 0;
  clock_t start;
  double timeEdlib = 0;
  double timeSimple = 0;

  for (int i = 0; i < numTests; i++) {
    bool failed = false;
    int queryLength = 500 + rand() % 300;
    int targetLength = 50000 + rand() % 10000;
    Element *query =
        static_cast<Element *>(malloc(sizeof(Element) * queryLength));
    Element *target =
        static_cast<Element *>(malloc(sizeof(Element) * targetLength));
    fillRandomly<Element, AlphabetIdx>(query, queryLength, alphabetSize);
    fillRandomly<Element, AlphabetIdx>(target, targetLength, alphabetSize);

    start = clock();
    EdlibAlignResult result = edlibAlign<Element, AlphabetIdx>(
        query, queryLength, target, targetLength,
        edlibNewAlignConfig<Element>(
            -1, mode, findAlignment ? EDLIB_TASK_PATH : EDLIB_TASK_DISTANCE,
            NULL, 0));
    timeEdlib += clock() - start;
    if (result.alignment) {
      if (!checkAlignment<Element>(query, queryLength, target,
                                   result.editDistance, result.endLocations[0],
                                   mode, result.alignment,
                                   result.alignmentLength)) {
        failed = true;
        printf("Alignment is not correct\n");
      }
      int alignmentStart = getAlignmentStart(
          result.alignment, result.alignmentLength, result.endLocations[0]);
      if (result.startLocations[0] != alignmentStart) {
        failed = true;
        printf(
            "Start location (%d) is not consistent with alignment start (%d)\n",
            result.startLocations[0], alignmentStart);
      }
    }

    start = clock();
    int score2;
    int numLocations2;
    int *endLocations2;
    calcEditDistanceSimple<Element>(query, queryLength, target, targetLength,
                                    mode, &score2, &endLocations2,
                                    &numLocations2);
    timeSimple += clock() - start;

    // Compare results
    if (result.editDistance != score2) {
      failed = true;
      printf("Scores are different! Expected %d, got %d)\n", score2,
             result.editDistance);
    } else if (result.editDistance == -1 && !(result.endLocations == NULL)) {
      failed = true;
      printf("Score was not found but endLocations is not NULL!\n");
    } else if (result.numLocations != numLocations2) {
      failed = true;
      printf(
          "Number of endLocations returned is not equal! Expected %d, got %d\n",
          numLocations2, result.numLocations);
    } else {
      for (int j = 0; j < result.numLocations; j++) {
        if (result.endLocations[j] != endLocations2[j]) {
          failed = true;
          printf("EndLocations at %d are not equal! Expected %d, got %d\n", j,
                 endLocations2[j], result.endLocations[j]);
          break;
        }
      }
    }

    edlibFreeAlignResult(result);
    if (endLocations2)
      delete[] endLocations2;

    for (int k = max(score2 - 1, 0); k <= score2 + 1; k++) {
      int scoreExpected = score2 > k ? -1 : score2;
      EdlibAlignResult result3 = edlibAlign<Element, AlphabetIdx>(
          query, queryLength, target, targetLength,
          edlibNewAlignConfig<Element>(
              k, mode, findAlignment ? EDLIB_TASK_PATH : EDLIB_TASK_DISTANCE,
              NULL, 0));
      if (result3.editDistance != scoreExpected) {
        failed = true;
        printf("For k = %d score was %d but it should have been %d\n", k,
               result3.editDistance, scoreExpected);
      }
      if (result3.alignment) {
        if (!checkAlignment<Element>(
                query, queryLength, target, result3.editDistance,
                result3.endLocations[0], mode, result3.alignment,
                result3.alignmentLength)) {
          failed = true;
          printf("Alignment is not correct\n");
        }
        int alignmentStart =
            getAlignmentStart(result3.alignment, result3.alignmentLength,
                              result3.endLocations[0]);
        if (result3.startLocations[0] != alignmentStart) {
          failed = true;
          printf("Start location (%d) is not consistent with alignment start "
                 "(%d)\n",
                 result3.startLocations[0], alignmentStart);
        }
      }
      edlibFreeAlignResult(result3);
    }

    if (failed)
      numTestsFailed++;

    free(query);
    free(target);
  }

  printf(mode == EDLIB_MODE_HW ? "HW: "
                               : mode == EDLIB_MODE_SHW ? "SHW: " : "NW: ");
  printf(numTestsFailed == 0 ? "\x1B[32m" : "\x1B[31m");
  printf("%d/%d", numTests - numTestsFailed, numTests);
  printf("\x1B[0m");
  printf(" random tests passed!\n");
  double mTime = static_cast<double>(timeEdlib) / CLOCKS_PER_SEC;
  double sTime = static_cast<double>(timeSimple) / CLOCKS_PER_SEC;
  printf("Time Edlib: %lf\n", mTime);
  printf("Time Simple: %lf\n", sTime);
  printf("Times faster: %.2lf\n", sTime / mTime);
  return numTestsFailed == 0;
}

TEST(EdlibTest, TestEmpty) {
  const char *emptySeq = "";
  const char *nonEmptySeq = "ACTG";
  const int nonEmptySeqLength = 4;

  for (auto mode :
       vector<EdlibAlignMode>{EDLIB_MODE_NW, EDLIB_MODE_HW, EDLIB_MODE_SHW}) {
    EXPECT_TRUE(executeTest(emptySeq, 0, nonEmptySeq, nonEmptySeqLength, mode));
    EXPECT_TRUE(executeTest(nonEmptySeq, nonEmptySeqLength, emptySeq, 0, mode));
  }
}

TEST(EdlibTest, TestBasic1) {
  int queryLength = 4;
  int targetLength = 4;
  char query[4] = {0, 1, 2, 3};
  char target[4] = {0, 1, 2, 3};

  for (auto mode :
       vector<EdlibAlignMode>{EDLIB_MODE_NW, EDLIB_MODE_HW, EDLIB_MODE_SHW}) {
    EXPECT_TRUE(executeTest(query, queryLength, target, targetLength, mode));
  }
}

TEST(EdlibTest, TestBasic2) {
  int queryLength = 5;
  int targetLength = 9;
  char query[5] = {0, 1, 2, 3, 4};              // "match"
  char target[9] = {8, 5, 0, 1, 3, 4, 6, 7, 5}; // "remachine"

  for (auto mode :
       vector<EdlibAlignMode>{EDLIB_MODE_NW, EDLIB_MODE_HW, EDLIB_MODE_SHW}) {
    EXPECT_TRUE(executeTest(query, queryLength, target, targetLength, mode));
  }
}

TEST(EdlibTest, TestBasic3) {
  int queryLength = 5;
  int targetLength = 9;
  char query[5] = {0, 1, 2, 3, 4};
  char target[9] = {1, 2, 0, 1, 2, 3, 4, 5, 4};

  for (auto mode :
       vector<EdlibAlignMode>{EDLIB_MODE_NW, EDLIB_MODE_HW, EDLIB_MODE_SHW}) {
    EXPECT_TRUE(executeTest(query, queryLength, target, targetLength, mode));
  }
}

TEST(EdlibTest, TestBasic4) {
  int queryLength = 200;
  int targetLength = 200;
  char query[200] = {0};
  char target[200] = {1};

  for (auto mode :
       vector<EdlibAlignMode>{EDLIB_MODE_NW, EDLIB_MODE_HW, EDLIB_MODE_SHW}) {
    EXPECT_TRUE(executeTest(query, queryLength, target, targetLength, mode));
  }
}

TEST(EdlibTest, TestBasic5) {
  int queryLength =
      64; // Testing for special case when queryLength == word size
  int targetLength = 64;
  char query[64] = {0};
  char target[64] = {1};

  for (auto mode :
       vector<EdlibAlignMode>{EDLIB_MODE_NW, EDLIB_MODE_HW, EDLIB_MODE_SHW}) {
    EXPECT_TRUE(executeTest(query, queryLength, target, targetLength, mode));
  }
}

TEST(EdlibTest, TestBasic6) {
  int queryLength =
      13; // Testing for special case when queryLength == word size
  int targetLength = 420;
  char query[13] = {1, 3, 0, 1, 1, 1, 3, 0, 1, 3, 1, 3, 3};
  char target[420] = {
      0, 1, 1, 1, 0, 1, 3, 0, 1, 3, 3, 3, 1, 3, 2, 2, 3, 2, 3, 3, 1, 0, 1, 1, 1,
      0, 1, 3, 0, 1, 3, 3, 3, 1, 3, 2, 2, 3, 2, 3, 3, 1, 0, 1, 1, 1, 0, 1, 3, 0,
      1, 3, 3, 3, 1, 3, 2, 2, 3, 2, 3, 3, 1, 0, 1, 1, 1, 0, 1, 3, 0, 1, 3, 3, 3,
      1, 3, 2, 2, 3, 2, 3, 3, 1, 0, 1, 1, 1, 0, 1, 3, 0, 1, 3, 3, 3, 1, 3, 2, 2,
      3, 2, 3, 3, 1, 0, 1, 1, 1, 0, 1, 3, 0, 1, 3, 3, 3, 1, 3, 2, 2, 3, 2, 3, 3,
      1, 0, 1, 1, 1, 0, 1, 3, 0, 1, 3, 3, 3, 1, 3, 2, 2, 3, 2, 3, 3, 1, 0, 1, 1,
      1, 0, 1, 3, 0, 1, 3, 3, 3, 1, 3, 2, 2, 3, 2, 3, 3, 1, 0, 1, 1, 1, 0, 1, 3,
      0, 1, 3, 3, 3, 1, 3, 2, 2, 3, 2, 3, 3, 1, 0, 1, 1, 1, 0, 1, 3, 0, 1, 3, 3,
      3, 1, 3, 2, 2, 3, 2, 3, 3, 1, 0, 1, 1, 1, 0, 1, 3, 0, 1, 3, 3, 3, 1, 3, 2,
      2, 3, 2, 3, 3, 1, 0, 1, 1, 1, 0, 1, 3, 0, 1, 3, 3, 3, 1, 3, 2, 2, 3, 2, 3,
      3, 1, 0, 1, 1, 1, 0, 1, 3, 0, 1, 3, 3, 3, 1, 3, 2, 2, 3, 2, 3, 3, 1, 0, 1,
      1, 1, 0, 1, 3, 0, 1, 3, 3, 3, 1, 3, 2, 2, 3, 2, 3, 3, 1, 0, 1, 1, 1, 0, 1,
      3, 0, 1, 3, 3, 3, 1, 3, 2, 2, 3, 2, 3, 3, 1, 0, 1, 1, 1, 0, 1, 3, 0, 1, 3,
      3, 3, 1, 3, 2, 2, 3, 2, 3, 3, 1, 0, 1, 1, 1, 0, 1, 3, 0, 1, 3, 3, 3, 1, 3,
      2, 2, 3, 2, 3, 3, 1, 0, 1, 1, 1, 0, 1, 3, 0, 1, 3, 3, 3, 1, 3, 2, 2, 3, 2,
      3, 3, 1, 0, 1, 1, 1, 0, 1, 3, 0, 1, 3, 3, 3, 1, 3, 2, 2, 3, 2, 3, 3, 1, 0,
      1, 1, 1, 0, 1, 3, 0, 1, 3, 3, 3, 1, 3, 2, 2, 3, 2, 3, 3, 1};

  for (auto mode :
       vector<EdlibAlignMode>{EDLIB_MODE_NW, EDLIB_MODE_HW, EDLIB_MODE_SHW}) {
    EXPECT_TRUE(executeTest(query, queryLength, target, targetLength, mode));
  }
}

TEST(EdlibTest, TestBasic7) {
  int queryLength = 3;
  int targetLength = 5;
  char query[3] = {2, 3, 0};
  char target[5] = {0, 1, 2, 2, 0};

  for (auto mode :
       vector<EdlibAlignMode>{EDLIB_MODE_NW, EDLIB_MODE_HW, EDLIB_MODE_SHW}) {
    EXPECT_TRUE(executeTest(query, queryLength, target, targetLength, mode));
  }
}

TEST(EdlibTest, TestBasic8) {
  int queryLength = 3;
  int targetLength = 3;
  char query[3] = {2, 3, 0};
  char target[3] = {2, 2, 0};

  for (auto mode :
       vector<EdlibAlignMode>{EDLIB_MODE_NW, EDLIB_MODE_HW, EDLIB_MODE_SHW}) {
    EXPECT_TRUE(executeTest(query, queryLength, target, targetLength, mode));
  }
}

TEST(EdlibTest, TestBasic9) {
  int queryLength = 64;
  int targetLength = 393;
  char query[64] = {9, 5, 5, 9, 9, 4, 6, 0, 1, 1, 5, 4, 6, 0, 6, 5,
                    5, 6, 5, 2, 2, 0, 6, 0, 8, 3, 7, 0, 6, 6, 4, 8,
                    3, 1, 9, 4, 5, 5, 5, 7, 8, 2, 3, 6, 4, 1, 1, 2,
                    7, 7, 6, 0, 9, 2, 0, 9, 6, 9, 9, 4, 6, 5, 2, 9};
  char target[393] = {
      7, 1, 6, 2, 9, 1, 1, 7, 5, 5, 4, 9, 6, 7, 3, 4, 6, 9, 4, 5, 2, 6, 6, 0, 7,
      8, 4, 3, 3, 9, 5, 2, 0, 1, 7, 1, 4, 0, 9, 9, 7, 5, 0, 6, 2, 4, 0, 9, 3, 6,
      6, 7, 4, 3, 9, 3, 3, 4, 7, 8, 5, 4, 1, 7, 7, 0, 9, 3, 0, 8, 4, 0, 3, 4, 6,
      7, 0, 8, 6, 6, 6, 5, 5, 2, 0, 5, 5, 3, 1, 4, 1, 6, 8, 4, 3, 7, 6, 2, 0, 9,
      0, 4, 9, 5, 1, 5, 3, 1, 3, 1, 9, 9, 6, 5, 1, 8, 0, 6, 1, 1, 1, 5, 9, 1, 1,
      2, 1, 8, 5, 1, 7, 7, 8, 6, 5, 9, 1, 0, 2, 4, 1, 2, 5, 0, 9, 6, 8, 1, 4, 2,
      4, 5, 9, 3, 9, 0, 5, 0, 8, 0, 3, 7, 0, 1, 3, 5, 0, 6, 5, 5, 2, 8, 9, 7, 0,
      8, 5, 1, 9, 0, 3, 3, 7, 2, 6, 6, 4, 3, 8, 5, 6, 2, 2, 6, 5, 8, 3, 8, 4, 0,
      3, 7, 8, 2, 6, 9, 0, 2, 0, 1, 2, 5, 6, 1, 9, 4, 8, 3, 7, 8, 8, 5, 2, 3, 1,
      8, 1, 6, 6, 7, 6, 9, 6, 5, 3, 3, 6, 5, 7, 8, 6, 1, 3, 4, 2, 4, 0, 0, 7, 7,
      1, 8, 5, 3, 3, 6, 1, 4, 5, 7, 3, 1, 8, 0, 8, 1, 5, 6, 6, 2, 4, 4, 3, 9, 8,
      7, 3, 8, 0, 3, 8, 1, 3, 3, 4, 6, 1, 8, 2, 6, 7, 5, 8, 6, 7, 8, 7, 4, 5, 6,
      6, 9, 0, 1, 1, 1, 9, 4, 9, 1, 9, 9, 2, 2, 4, 8, 0, 6, 6, 4, 4, 4, 2, 2, 2,
      9, 3, 1, 6, 8, 7, 2, 9, 8, 6, 0, 1, 7, 7, 2, 8, 6, 2, 2, 1, 6, 0, 3, 4, 9,
      8, 9, 3, 2, 3, 5, 3, 6, 6, 9, 6, 6, 2, 6, 6, 0, 8, 7, 9, 5, 9, 7, 4, 3, 1,
      7, 2, 1, 0, 6, 0, 0, 7, 5, 2, 1, 2, 6, 9, 1, 5, 6, 7};

  for (auto mode :
       vector<EdlibAlignMode>{EDLIB_MODE_NW, EDLIB_MODE_HW, EDLIB_MODE_SHW}) {
    EXPECT_TRUE(executeTest(query, queryLength, target, targetLength, mode));
  }
}

TEST(EdlibTest, TestBasic10) {
  int queryLength = 3;
  int targetLength = 3;
  char query[3] = {0, 1, 2};
  char target[3] = {1, 1, 1};

  for (auto mode :
       vector<EdlibAlignMode>{EDLIB_MODE_NW, EDLIB_MODE_HW, EDLIB_MODE_SHW}) {
    EXPECT_TRUE(executeTest(query, queryLength, target, targetLength, mode));
  }
}

TEST(EdlibTest, TestBasic11) {
  int queryLength = 8;
  int targetLength = 8;
  // NOTE(Martin): I am using CHAR_MIN and CHAR_MAX because 'char' type is not
  // guaranteed to be
  //   signed or unsigned by compiler, we can't know if it is signed or
  //   unsigned.
  char query[8] = {CHAR_MIN, CHAR_MIN + (CHAR_MAX - CHAR_MIN) / 2, CHAR_MAX};
  char target[8] = {CHAR_MIN, CHAR_MIN + (CHAR_MAX - CHAR_MIN) / 2 + 1,
                    CHAR_MAX};

  for (auto mode :
       vector<EdlibAlignMode>{EDLIB_MODE_NW, EDLIB_MODE_HW, EDLIB_MODE_SHW}) {
    EXPECT_TRUE(executeTest(query, queryLength, target, targetLength, mode));
  }
}

TEST(EdlibTest, TestBasic12) {
  EdlibEqualityPair<char> additionalEqualities[24] = {
      {'R', 'A'}, {'R', 'G'}, {'M', 'A'}, {'M', 'C'}, {'W', 'A'}, {'W', 'T'},
      {'S', 'C'}, {'S', 'G'}, {'Y', 'C'}, {'Y', 'T'}, {'K', 'G'}, {'K', 'T'},
      {'V', 'A'}, {'V', 'C'}, {'V', 'G'}, {'H', 'A'}, {'H', 'C'}, {'H', 'T'},
      {'D', 'A'}, {'D', 'G'}, {'D', 'T'}, {'B', 'C'}, {'B', 'G'}, {'B', 'T'}};
  const char *query = "GCATATCAATAAGCGGAGGA";
  const char *target =
      "TAACAAGGTTTCCGTAGGTGAACCTGCGGAAGGATCATTATCGAATAAACTTGATGGGTTGTCGCTGGCTTC"
      "TAGGAGCATGTGCACATCCGTCATTTTTATCCATCCACCTGTGCACCTTTTGTAGTCTTTGGAGGTAATAAG"
      "CGTGAATCTATCGAGGTCCTCTGGTCCTCGGAAAGAGGTGTTTGCCATATGGCTCGCCTTTGATACTCGCGA"
      "GTTACTCTAAGACTATGTCCTTTCATATACTACGAATGTAATAGAATGTATTCATTGGGCCTCAGTGCCTAT"
      "AAAACATATACAACTTTCAGCAACGGATCTCTTGGCTCTCGCATCGATGAAGAACGCAGCGAAATGCGATAA"
      "GTAATGTGAATTGCAGAATTCAGTGAATCATCGAATCTTTGAACGCACCTTGCGCTCCTTGGTATTCCGAGG"
      "AGCATGCCTGTTTGAGTGTCATTAAATTCTCAACCCCTTCCGGTTTTTTGACTGGCTTTGGGGCTTGGATGT"
      "GGGGGATTCATTTGCGGGCCTCTGTAGAGGTCGGCTCCCCTGAAATGCATTAGTGGAACCGTTTGCGGTTAC"
      "CGTCGCTGGTGTGATAACTATCTATGCCAAAGACAAACTGCTCTCTGATAGTTCTGCTTCTAACCGTCCATT"
      "TATTGGACAACATTATTATGAACACTTGACCTCAAATCAGGTAGGACTACCCGCTGAACTTAAGCATATCAA"
      "TAAGCGGAGGAAAAGAAACTAACAAGGATTCCCCTAGTAACTGCGAGTGAAGCGGGAAAAGCTCAAATTTAA"
      "AATCTGGCGGTCTTTGGCCGTCCGAGTTGTAATCTAGAGAAGCGACACCCGCGCTGGACCGTGTACAAGTCT"
      "CCTGGAATGGAGCGTCATAGAGGGTGAGAATCCCGTCTCTGACACGGACTACCAGGGCTTTGTGGTGCGCTC"
      "TCAAAGAGTCGAGTTGTTTGGGAATGCAGCTCTAAATGGGTGGTAAATTCCATCTAAAGCTAAATATTGGCG"
      "AGAGACCGATAGCGAACAAGTACCGTGAGGGAAAGATGAAAAGAACTTTGGAAAGAGAGTTAAACAGTACGT"
      "GAAATTGCTGAAAGGGAAACGCTTGAAGTCAGTCGCGTTGGCCGGGGATCAGCCTCGCTTTTGCGTGGTGTA"
      "TTTCCTGGTTGACGGGTCAGCATCAATTTTGACCGCTGGAAAAGGACTTGGGGAATGTGGCATCTTCGGATG"
      "TGTTATAGCCCTTTGTCGCATACGGCGGTTGGGATTGAGGAACTCAGCACGCCGCAAGGCCGGGTTTCGACC"
      "ACGTTCGTGCTTAGGATGCTGGCATAATGGCTTTAATCGACCCGTCTTGAAACACGGACCAAGGAGTCTAAC"
      "ATGCCTGCGAGTGTTTGGGTGGAAAACCCGAGCGCGTAATGAAAGTGAAAGTTGAGATCCCTGTCGTGGGGA"
      "GCATCGACGCCCGGACCAGAACTTTTGGGACGGATCTGCGGTAGAGCATGTATGTTGGGACCCGAAAGATGG"
      "TGAACTATGCCTGAATAGGGTGAAGCCAGAGGAAACTCTGGTGGAGGCTCGTAGCGATTCTGACGTGCAAAT"
      "CGATCGTCAAATTTGGGTATAGGGGCGAAAGACTAATCGAACCATCTAGTAGCTGGTTCCTGCCGAAGTTTC"
      "CCTCAGGATAGCAGAAACTCATATCAGATTTATGTGGTAAAGCGAATGATTAGAGGCCTTGGGGTTGAAACA"
      "ACCTTAACCTATTCTCAAACTTTAAATATGTAAGAACGAGCCGTTTCTTGATTGAACCGCTCGGCGATTGAG"
      "AGTTTCTAGTGGGCCATTTTTGGTAAGCAGAACTGGCGATGCGGGATGAACCGAACGCGAGGTTAAGGTGCC"
      "GGAATTCACGCTCATCAGACACCACAAAAGGTGTTAGTTCATCTAGACAGCAGGACGGTGGCCATGGAAGTC"
      "GGAATCCGCTAAGGAGTGTGTAACAACTCACCTGCCGAATGAACTAGCCCTGAAAATGGATGGCGCTTAAGC"
      "GTGATACCCATACCTCGCCGTCAGCGTTGAAGTGACGCGCTGACGAGTAGGCAGGCGTGGAGGTCAGTGAAG"
      "AAGCCTTGGCAGTGATGCTGGGTGAAACGGCCTCC";

  EdlibAlignResult result =
      edlibAlign(query, static_cast<int>(std::strlen(query)), target,
                 static_cast<int>(std::strlen(target)),
                 edlibNewAlignConfig<char>(-1, EDLIB_MODE_HW, EDLIB_TASK_LOC,
                                           additionalEqualities, 24));
  bool pass = result.status == EDLIB_STATUS_OK && result.editDistance == 0;
  edlibFreeAlignResult(result);
  EXPECT_TRUE(pass);
}

TEST(EdlibTest, TestBasic13) {
  // In this test, one of optimal solutions is:
  //         B
  //       AA
  // which brings us into interesting situation where one of end locations is
  // -1.
  const char *query = "AA";
  const char *target = "B";

  EdlibAlignResult result = edlibAlign(
      query, static_cast<int>(std::strlen(query)), target,
      static_cast<int>(std::strlen(target)),
      edlibNewAlignConfig<char>(-1, EDLIB_MODE_HW, EDLIB_TASK_PATH, NULL, 0));
  bool pass = result.status == EDLIB_STATUS_OK && result.editDistance == 2;
  edlibFreeAlignResult(result);
  EXPECT_TRUE(pass);
}

TEST(EdlibTest, TestBasic14) {
  // In this test, one of optimal solutions is:
  //         B
  //       AA
  // which brings us into interesting situation where one of end locations is
  // -1.
  const char *query = "AA";
  const char *target = "B";

  EdlibAlignResult result = edlibAlign(
      query, static_cast<int>(std::strlen(query)), target,
      static_cast<int>(std::strlen(target)),
      edlibNewAlignConfig<char>(-1, EDLIB_MODE_SHW, EDLIB_TASK_PATH, NULL, 0));
  bool pass = result.status == EDLIB_STATUS_OK && result.editDistance == 2;
  edlibFreeAlignResult(result);
  EXPECT_TRUE(pass);
}

TEST(EdlibTest, TestBasic15) {
  // In this test, optimal alignment is when query and target overlap, query end
  // with target start, HW.
  const char *query = "AAABBB";
  const char *target = "BBBC";

  EdlibAlignResult result = edlibAlign(
      query, static_cast<int>(std::strlen(query)), target,
      static_cast<int>(std::strlen(target)),
      edlibNewAlignConfig<char>(-1, EDLIB_MODE_HW, EDLIB_TASK_LOC, NULL, 0));
  bool pass = result.status == EDLIB_STATUS_OK && result.editDistance == 3;
  edlibFreeAlignResult(result);
  EXPECT_TRUE(pass);
}

TEST(EdlibTest, TestBasic16) {
  // In this test, optimal alignment is when query and target overlap, query
  // start with target end, HW.
  const char *query = "BBBAAA";
  const char *target = "CBBB";

  EdlibAlignResult result = edlibAlign(
      query, static_cast<int>(std::strlen(query)), target,
      static_cast<int>(std::strlen(target)),
      edlibNewAlignConfig<char>(-1, EDLIB_MODE_HW, EDLIB_TASK_LOC, NULL, 0));
  bool pass = result.status == EDLIB_STATUS_OK && result.editDistance == 3;
  edlibFreeAlignResult(result);
  EXPECT_TRUE(pass);
}

TEST(EdlibTest, TestCigar) {
  unsigned char alignment[] = {
      EDLIB_EDOP_MATCH,  EDLIB_EDOP_MATCH,  EDLIB_EDOP_INSERT,
      EDLIB_EDOP_INSERT, EDLIB_EDOP_INSERT, EDLIB_EDOP_DELETE,
      EDLIB_EDOP_INSERT, EDLIB_EDOP_INSERT, EDLIB_EDOP_MISMATCH,
      EDLIB_EDOP_MATCH,  EDLIB_EDOP_MATCH};
  char *cigar = edlibAlignmentToCigar(alignment, 11, EDLIB_CIGAR_EXTENDED);
  bool pass = true;
  char expected[] = "2=3I1D2I1X2=";
  if (strcmp(cigar, expected) != 0) {
    pass = false;
    printf("Expected %s, got %s\n", expected, cigar);
  }

  if (cigar)
    free(cigar);

  EXPECT_TRUE(pass);

  cigar = edlibAlignmentToCigar(alignment, 11, EDLIB_CIGAR_STANDARD);
  pass = true;
  char expected2[] = "2M3I1D2I3M";
  if (strcmp(cigar, expected2) != 0) {
    pass = false;
    printf("Expected %s, got %s\n", expected2, cigar);
  }
  if (cigar)
    free(cigar);

  EXPECT_TRUE(pass);
}

TEST(EdlibTest, TestCustomEqualityRelation) {
  EdlibEqualityPair<char> additionalEqualities[6] = {
      {'R', 'A'}, {'R', 'G'}, {'N', 'A'}, {'N', 'C'}, {'N', 'T'}, {'N', 'G'}};

  const char *query = "GTGNRTCARCGAANCTTTN";
  const char *target = "GTGAGTCATCGAATCTTTGAACGCACCTTGCGCTCCTTGGT";

  EdlibAlignResult result =
      edlibAlign(query, 19, target, 41,
                 edlibNewAlignConfig<char>(-1, EDLIB_MODE_HW, EDLIB_TASK_PATH,
                                           additionalEqualities, 6));
  bool pass = result.status == EDLIB_STATUS_OK && result.editDistance == 1;
  edlibFreeAlignResult(result);
  EXPECT_TRUE(pass);
}

TEST(EdlibTest, TestRandom) {
  srand(42);
  int numRandomTests = 10;
  for (auto mode : vector<EdlibAlignMode>{
           /*EDLIB_MODE_NW,*/ EDLIB_MODE_HW /*, EDLIB_MODE_SHW*/}) {
    printf("char, vocab size : 100\n");
    EXPECT_TRUE(
        (runRandomTests<char, unsigned char>(numRandomTests, mode, true, 100)));
    EXPECT_TRUE((
        runRandomTests<char, unsigned char>(numRandomTests, mode, false, 100)));
    printf("int16_t, vocab size : 100\n");
    EXPECT_TRUE(
        (runRandomTests<int16_t, uint16_t>(numRandomTests, mode, true, 100)));
    EXPECT_TRUE(
        (runRandomTests<int16_t, uint16_t>(numRandomTests, mode, false, 100)));
    printf("int16_t, vocab size : 10000\n");
    EXPECT_TRUE(
        (runRandomTests<int16_t, uint16_t>(numRandomTests, mode, true, 10000)));
    EXPECT_TRUE((
        runRandomTests<int16_t, uint16_t>(numRandomTests, mode, false, 10000)));
    printf("int32_t, vocab size : 100\n");
    EXPECT_TRUE(
        (runRandomTests<int32_t, uint32_t>(numRandomTests, mode, true, 100)));
    EXPECT_TRUE(
        (runRandomTests<int32_t, uint32_t>(numRandomTests, mode, false, 100)));
    printf("int32_t, vocab size : 10000\n");
    EXPECT_TRUE(
        (runRandomTests<int32_t, uint32_t>(numRandomTests, mode, true, 10000)));
    EXPECT_TRUE((
        runRandomTests<int32_t, uint32_t>(numRandomTests, mode, false, 10000)));
  }
}
