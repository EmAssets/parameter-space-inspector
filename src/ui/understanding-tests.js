// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// The live understanding battery: one source for the Understanding view (understand.js) and the Tipping-point
// scorecard (src/lesion.py reads this file: the value after the "=" below, up to the closing bracket, is strict JSON).
// self_referential: the test rewards this framework's own vocabulary, so it is left out of the headline count.
window.PSI_UNDERSTANDING_TESTS = [
  {"id": "agree-keys", "what": "agreement across a distractor", "prompt": "The keys to the old cabinet", "good": [" are"], "bad": [" is"]},
  {"id": "agree-price", "what": "agreement across a distractor", "prompt": "The price of the apples", "good": [" has"], "bad": [" have"]},
  {"id": "conseq-glass", "what": "consequence (physics)", "prompt": "She dropped the glass on the stone floor. It", "good": [" shattered", " broke"], "bad": [" bounced", " flew"]},
  {"id": "conseq-icecream", "what": "consequence (everyday)", "prompt": "He left the ice cream in the sun. It", "good": [" melted"], "bad": [" froze"]},
  {"id": "state-coat", "what": "state tracking", "prompt": "Ravi put his keys in his coat, then left the coat at home. His keys are in the", "good": [" coat"], "bad": [" car"]},
  {"id": "state-book", "what": "state tracking", "prompt": "Tom gave the book to Anna. Anna gave it to Sam. Now the book is with", "good": [" Sam"], "bad": [" Tom"]},
  {"id": "novel-zorblax", "what": "a word it has never seen", "prompt": "In the old market there is a stall everyone calls the zorblax. Early each morning the zorblax", "good": [" opens", " sells"], "bad": [" sleeps", " swims"]},
  {"id": "street-kolkata", "what": "the street (knowledge of a place)", "prompt": "In Kolkata, the yellow vehicles that take passengers across the city are", "good": [" taxis", " cabs"], "bad": [" buses", " boats"]},
  {"id": "levels-market", "what": "levels (the Emergence Machine)", "prompt": "A market price is set by no single trader; it emerges from the interaction of many. On the ladder of levels, that makes a market price a property of", "good": [" institutions", " markets", " society", " systems"], "bad": [" atoms", " cells"], "self_referential": true, "note": "self-referential: rewards this framework's own words"}
];
