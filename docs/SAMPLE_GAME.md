# Self-play game (seed 2026)

**P0: net300(models/imitation_v5.pt)** plays Rhodes; **P1: net300(models/imitation_v5.pt)** plays Ephesus.  
Face-up Progress tokens: ['Urbanism', 'Culture', 'Propaganda'].  Conflict tokens: 3.


## Turn 1 — P0 to move

Table: P0 deck[25]: shield_h1 | P1 deck[25]: shield_h2 | center[60]: ?  
P0 Rhodes stage 0/5  score  0  shields 0  mil 0  cat no  tokens []
       cards: -  
P1 Ephesus stage 0/5  score  0  shields 0  mil 0  cat no  tokens []
       cards: -  
Conflict: 0/3, face-up tokens ['Urbanism', 'Culture', 'Propaganda']

- P0 pick a card (main) → **pick_center**  search: root value +0.23; pick_left(own deck) n=49 q=+0.19, pick_right(opp deck) n=50 q=+0.19, pick_center n=201 q=+0.24

## Turn 2 — P1 to move

Table: P0 deck[25]: shield_h1 | P1 deck[25]: shield_h2 | center[59]: ?  
P0 Rhodes stage 0/5  score  4  shields 0  mil 0  cat yes  tokens []
       cards: civ2catx1  
P1 Ephesus stage 0/5  score  0  shields 0  mil 0  cat no  tokens []
       cards: -  
Conflict: 0/3, face-up tokens ['Urbanism', 'Culture', 'Propaganda']

- P1 pick a card (main) → **pick_center**  search: root value -0.11; pick_right(opp deck) n=29 q=-0.16, pick_left(own deck) n=30 q=-0.15, pick_center n=241 q=-0.10

## Turn 3 — P0 to move

Table: P0 deck[25]: shield_h1 | P1 deck[25]: shield_h2 | center[58]: compass (known to [0])  
P0 Rhodes stage 0/5  score  4  shields 0  mil 0  cat yes  tokens []
       cards: civ2catx1  
P1 Ephesus stage 0/5  score  0  shields 1  mil 0  cat no  tokens []
       cards: shieldx1  
Conflict: 0/3, face-up tokens ['Urbanism', 'Culture', 'Propaganda']

- P0 pick a card (main) → **pick_center**  search: root value -0.23; pick_left(own deck) n=27 q=-0.19, pick_right(opp deck) n=44 q=-0.15, pick_center n=229 q=-0.25

## Turn 4 — P1 to move

Table: P0 deck[25]: shield_h1 | P1 deck[25]: shield_h2 | center[57]: ?  
P0 Rhodes stage 0/5  score  4  shields 0  mil 0  cat yes  tokens []
       cards: civ2catx1, compassx1  
P1 Ephesus stage 0/5  score  0  shields 1  mil 0  cat no  tokens []
       cards: shieldx1  
Conflict: 0/3, face-up tokens ['Urbanism', 'Culture', 'Propaganda']

- P1 pick a card (main) → **pick_center**  search: root value +0.23; pick_right(opp deck) n=12 q=+0.01, pick_left(own deck) n=22 q=+0.08, pick_center n=266 q=+0.26

## Turn 5 — P0 to move

Table: P0 deck[25]: shield_h1 | P1 deck[25]: shield_h2 | center[56]: ?  
P0 Rhodes stage 0/5  score  2  shields 0  mil 0  cat no  tokens []
       cards: civ2catx1, compassx1  
P1 Ephesus stage 0/5  score  4  shields 1  mil 0  cat yes  tokens []
       cards: civ2catx1, shieldx1  
Conflict: 0/3, face-up tokens ['Urbanism', 'Culture', 'Propaganda']

- P0 pick a card (main) → **pick_center**  search: root value -0.10; pick_left(own deck) n=41 q=-0.14, pick_right(opp deck) n=73 q=-0.08, pick_center n=186 q=-0.10
- P0 choose a Progress token (science); face-up: ['Urbanism', 'Culture', 'Propaganda'] → **token_Urbanism**  search: root value +0.13; token_Propaganda n=0, token_Urbanism n=311 q=+0.13, token_Culture n=0, token_blind n=0

## Turn 6 — P1 to move

Table: P0 deck[25]: shield_h1 | P1 deck[25]: shield_h2 | center[55]: papyrus (known to [1])  
P0 Rhodes stage 0/5  score  2  shields 0  mil 0  cat no  tokens ['Urbanism']
       cards: civ2catx1  
P1 Ephesus stage 0/5  score  4  shields 1  mil 0  cat yes  tokens []
       cards: civ2catx1, shieldx1  
Conflict: 0/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P1 pick a card (main) → **pick_center**  search: root value -0.11; pick_right(opp deck) n=6 q=-0.35, pick_left(own deck) n=8 q=-0.28, pick_center n=286 q=-0.10

## Turn 7 — P0 to move

Table: P0 deck[25]: shield_h1 | P1 deck[25]: shield_h2 | center[54]: ?  
P0 Rhodes stage 0/5  score  2  shields 0  mil 0  cat no  tokens ['Urbanism']
       cards: civ2catx1  
P1 Ephesus stage 0/5  score  4  shields 1  mil 0  cat yes  tokens []
       cards: papyrusx1, civ2catx1, shieldx1  
Conflict: 0/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P0 pick a card (main) → **pick_center**  search: root value +0.09; pick_left(own deck) n=50 q=+0.07, pick_right(opp deck) n=51 q=+0.09, pick_center n=199 q=+0.09

## Turn 8 — P1 to move

Table: P0 deck[25]: shield_h1 | P1 deck[25]: shield_h2 | center[53]: coin (known to [1])  
P0 Rhodes stage 0/5  score  2  shields 0  mil 0  cat no  tokens ['Urbanism']
       cards: papyrusx1, civ2catx1  
P1 Ephesus stage 0/5  score  4  shields 1  mil 0  cat yes  tokens []
       cards: papyrusx1, civ2catx1, shieldx1  
Conflict: 0/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P1 pick a card (main) → **pick_center**  search: root value -0.13; pick_right(opp deck) n=1 q=-0.08, pick_left(own deck) n=1 q=-0.40, pick_center n=298 q=-0.13

## Turn 9 — P0 to move

Table: P0 deck[25]: shield_h1 | P1 deck[25]: shield_h2 | center[51]: ?  
P0 Rhodes stage 0/5  score  2  shields 0  mil 0  cat no  tokens ['Urbanism']
       cards: papyrusx1, civ2catx1  
P1 Ephesus stage 1/5  score  9  shields 1  mil 0  cat yes  tokens []
       cards: civ3x1, civ2catx1, shieldx1  
Conflict: 0/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P0 pick a card (main) → **pick_center**  search: root value +0.16; pick_left(own deck) n=37 q=+0.15, pick_right(opp deck) n=19 q=+0.08, pick_center n=244 q=+0.17

## Turn 10 — P1 to move

Table: P0 deck[25]: shield_h1 | P1 deck[25]: shield_h2 | center[50]: shield_h2 (known to [1])  
P0 Rhodes stage 1/5  score  5  shields 0  mil 0  cat no  tokens ['Urbanism']
       cards: civ2catx1  
P1 Ephesus stage 1/5  score  9  shields 1  mil 0  cat yes  tokens []
       cards: civ3x1, civ2catx1, shieldx1  
Conflict: 0/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P1 pick a card (main) → **pick_center**  search: root value -0.12; pick_right(opp deck) n=70 q=-0.14, pick_left(own deck) n=98 q=-0.13, pick_center n=132 q=-0.10

## Turn 11 — P0 to move

Table: P0 deck[25]: shield_h1 | P1 deck[25]: shield_h2 | center[49]: ?  
P0 Rhodes stage 1/5  score  5  shields 0  mil 0  cat no  tokens ['Urbanism']
       cards: civ2catx1  
P1 Ephesus stage 1/5  score  9  shields 2  mil 0  cat yes  tokens []
       cards: civ3x1, civ2catx1, shieldx1, shield_h2x1  
Conflict: 2/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P0 pick a card (main) → **pick_center**  search: root value +0.09; pick_left(own deck) n=2 q=-0.17, pick_right(opp deck) n=2 q=-0.14, pick_center n=296 q=+0.10
- P0 pick a card (token:2, optional) → **pick_center**  search: root value +0.21; pick_left(own deck) n=2 q=-0.06, pick_right(opp deck) n=4 q=+0.07, pick_center n=308 q=+0.21, skip n=4 q=+0.10

## Turn 12 — P1 to move

Table: P0 deck[25]: shield_h1 | P1 deck[25]: shield_h2 | center[47]: shield_h1 (known to [1])  
P0 Rhodes stage 1/5  score  5  shields 0  mil 0  cat no  tokens ['Urbanism']
       cards: woodx1, civ2catx1, compassx1  
P1 Ephesus stage 1/5  score  9  shields 2  mil 0  cat yes  tokens []
       cards: civ3x1, civ2catx1, shieldx1, shield_h2x1  
Conflict: 2/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P1 pick a card (main) → **pick_left(own deck)**  search: root value -0.36; pick_right(opp deck) n=97 q=-0.35, pick_left(own deck) n=102 q=-0.37, pick_center n=101 q=-0.37

## Turn 13 — P0 to move

Table: P0 deck[25]: shield_h1 | P1 deck[24]: civ2cat | center[47]: ?  
P0 Rhodes stage 1/5  score  5  shields 0  mil 0  cat no  tokens ['Urbanism']
       cards: woodx1, civ2catx1, compassx1  
P1 Ephesus stage 1/5  score 15  shields 1  mil 2  cat yes  tokens []
       cards: civ3x1, civ2catx1, shieldx1  
Conflict: 0/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P0 pick a card (main) → **pick_center**  search: root value +0.41; pick_left(own deck) n=38 q=+0.44, pick_right(opp deck) n=44 q=+0.35, pick_center n=218 q=+0.42

## Turn 14 — P1 to move

Table: P0 deck[25]: shield_h1 | P1 deck[24]: civ2cat | center[46]: glass (known to [1])  
P0 Rhodes stage 1/5  score  5  shields 1  mil 0  cat no  tokens ['Urbanism']
       cards: woodx1, civ2catx1, compassx1, shield_h1x1  
P1 Ephesus stage 1/5  score 15  shields 1  mil 2  cat yes  tokens []
       cards: civ3x1, civ2catx1, shieldx1  
Conflict: 1/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P1 pick a card (main) → **pick_center**  search: root value -0.40; pick_right(opp deck) n=79 q=-0.52, pick_left(own deck) n=19 q=-0.40, pick_center n=202 q=-0.35

## Turn 15 — P0 to move

Table: P0 deck[25]: shield_h1 | P1 deck[24]: civ2cat | center[45]: ?  
P0 Rhodes stage 1/5  score  5  shields 1  mil 0  cat no  tokens ['Urbanism']
       cards: woodx1, civ2catx1, compassx1, shield_h1x1  
P1 Ephesus stage 1/5  score 15  shields 1  mil 2  cat yes  tokens []
       cards: glassx1, civ3x1, civ2catx1, shieldx1  
Conflict: 1/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P0 pick a card (main) → **pick_center**  search: root value +0.35; pick_left(own deck) n=94 q=+0.30, pick_right(opp deck) n=21 q=+0.37, pick_center n=185 q=+0.37

## Turn 16 — P1 to move

Table: P0 deck[25]: shield_h1 | P1 deck[24]: civ2cat | center[44]: coin (known to [1])  
P0 Rhodes stage 2/5  score  9  shields 2  mil 0  cat no  tokens ['Urbanism']
       cards: civ2catx1, compassx1, shield_h1x1  
P1 Ephesus stage 1/5  score 15  shields 1  mil 2  cat yes  tokens []
       cards: glassx1, civ3x1, civ2catx1, shieldx1  
Conflict: 1/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P1 pick a card (main) → **pick_center**  search: root value -0.48; pick_right(opp deck) n=29 q=-0.57, pick_left(own deck) n=2 q=-0.49, pick_center n=269 q=-0.47

## Turn 17 — P0 to move

Table: P0 deck[25]: shield_h1 | P1 deck[24]: civ2cat | center[43]: ?  
P0 Rhodes stage 2/5  score  9  shields 2  mil 0  cat no  tokens ['Urbanism']
       cards: civ2catx1, compassx1, shield_h1x1  
P1 Ephesus stage 2/5  score 19  shields 1  mil 2  cat yes  tokens []
       cards: civ3x1, civ2catx1, shieldx1  
Conflict: 1/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P0 pick a card (main) → **pick_right(opp deck)**  search: root value +0.47; pick_left(own deck) n=71 q=+0.48, pick_right(opp deck) n=121 q=+0.45, pick_center n=108 q=+0.47

## Turn 18 — P1 to move

Table: P0 deck[25]: shield_h1 | P1 deck[23]: wood | center[43]: ?  
P0 Rhodes stage 2/5  score 13  shields 2  mil 0  cat yes  tokens ['Urbanism']
       cards: civ2catx2, compassx1, shield_h1x1  
P1 Ephesus stage 2/5  score 17  shields 1  mil 2  cat no  tokens []
       cards: civ3x1, civ2catx1, shieldx1  
Conflict: 1/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P1 pick a card (main) → **pick_left(own deck)**  search: root value -0.45; pick_right(opp deck) n=17 q=-0.63, pick_left(own deck) n=264 q=-0.43, pick_center n=19 q=-0.54

## Turn 19 — P0 to move

Table: P0 deck[25]: shield_h1 | P1 deck[22]: papyrus | center[43]: shield (known to [0])  
P0 Rhodes stage 2/5  score 13  shields 2  mil 0  cat yes  tokens ['Urbanism']
       cards: civ2catx2, compassx1, shield_h1x1  
P1 Ephesus stage 2/5  score 17  shields 1  mil 2  cat no  tokens []
       cards: woodx1, civ3x1, civ2catx1, shieldx1  
Conflict: 1/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P0 pick a card (main) → **pick_center**  search: root value +0.48; pick_left(own deck) n=37 q=+0.47, pick_right(opp deck) n=71 q=+0.52, pick_center n=192 q=+0.47

## Turn 20 — P1 to move

Table: P0 deck[25]: shield_h1 | P1 deck[22]: papyrus | center[42]: ?  
P0 Rhodes stage 2/5  score 13  shields 3  mil 0  cat yes  tokens ['Urbanism']
       cards: civ2catx2, compassx1, shieldx1, shield_h1x1  
P1 Ephesus stage 2/5  score 17  shields 1  mil 2  cat no  tokens []
       cards: woodx1, civ3x1, civ2catx1, shieldx1  
Conflict: 1/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P1 pick a card (main) → **pick_left(own deck)**  search: root value -0.45; pick_right(opp deck) n=19 q=-0.54, pick_left(own deck) n=198 q=-0.45, pick_center n=83 q=-0.43

## Turn 21 — P0 to move

Table: P0 deck[25]: shield_h1 | P1 deck[21]: stone | center[42]: glass (known to [0])  
P0 Rhodes stage 2/5  score 13  shields 3  mil 0  cat yes  tokens ['Urbanism']
       cards: civ2catx2, compassx1, shieldx1, shield_h1x1  
P1 Ephesus stage 2/5  score 17  shields 1  mil 2  cat no  tokens []
       cards: woodx1, papyrusx1, civ3x1, civ2catx1, shieldx1  
Conflict: 1/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P0 pick a card (main) → **pick_center**  search: root value +0.10; pick_left(own deck) n=6 q=+0.00, pick_right(opp deck) n=133 q=+0.11, pick_center n=161 q=+0.09

## Turn 22 — P1 to move

Table: P0 deck[25]: shield_h1 | P1 deck[21]: stone | center[41]: ?  
P0 Rhodes stage 2/5  score 13  shields 3  mil 0  cat yes  tokens ['Urbanism']
       cards: glassx1, civ2catx2, compassx1, shieldx1, shield_h1x1  
P1 Ephesus stage 2/5  score 17  shields 1  mil 2  cat no  tokens []
       cards: woodx1, papyrusx1, civ3x1, civ2catx1, shieldx1  
Conflict: 1/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P1 pick a card (main) → **pick_left(own deck)**  search: root value -0.08; pick_right(opp deck) n=1 q=-0.54, pick_left(own deck) n=298 q=-0.08, pick_center n=1 q=-0.41

## Turn 23 — P0 to move

Table: P0 deck[25]: shield_h1 | P1 deck[20]: gear | center[40]: stone (known to [0])  
P0 Rhodes stage 2/5  score 13  shields 3  mil 0  cat yes  tokens ['Urbanism']
       cards: glassx1, civ2catx2, compassx1, shieldx1, shield_h1x1  
P1 Ephesus stage 3/5  score 21  shields 2  mil 2  cat no  tokens []
       cards: civ3x1, civ2catx1, shieldx2  
Conflict: 1/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P0 pick a card (main) → **pick_center**  search: root value -0.02; pick_left(own deck) n=29 q=-0.12, pick_right(opp deck) n=41 q=-0.13, pick_center n=230 q=+0.02

## Turn 24 — P1 to move

Table: P0 deck[25]: shield_h1 | P1 deck[20]: gear | center[39]: ?  
P0 Rhodes stage 2/5  score 13  shields 3  mil 0  cat yes  tokens ['Urbanism']
       cards: stonex1, glassx1, civ2catx2, compassx1, shieldx1, shield_h1x1  
P1 Ephesus stage 3/5  score 21  shields 2  mil 2  cat no  tokens []
       cards: civ3x1, civ2catx1, shieldx2  
Conflict: 1/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P1 pick a card (main) → **pick_center**  search: root value -0.00; pick_right(opp deck) n=27 q=-0.21, pick_left(own deck) n=54 q=-0.05, pick_center n=219 q=+0.03

## Turn 25 — P0 to move

Table: P0 deck[25]: shield_h1 | P1 deck[20]: gear | center[38]: coin (known to [0])  
P0 Rhodes stage 2/5  score 13  shields 3  mil 0  cat yes  tokens ['Urbanism']
       cards: stonex1, glassx1, civ2catx2, compassx1, shieldx1, shield_h1x1  
P1 Ephesus stage 3/5  score 21  shields 2  mil 2  cat no  tokens []
       cards: stonex1, civ3x1, civ2catx1, shieldx2  
Conflict: 1/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P0 pick a card (main) → **pick_center**  search: root value +0.10; pick_left(own deck) n=1 q=-0.16, pick_right(opp deck) n=1 q=-0.21, pick_center n=298 q=+0.11

## Turn 26 — P1 to move

Table: P0 deck[25]: shield_h1 | P1 deck[20]: gear | center[37]: ?  
P0 Rhodes stage 3/5  score 18  shields 3  mil 0  cat yes  tokens ['Urbanism']
       cards: civ2catx2, compassx1, shieldx1, shield_h1x1  
P1 Ephesus stage 3/5  score 21  shields 2  mil 2  cat no  tokens []
       cards: stonex1, civ3x1, civ2catx1, shieldx2  
Conflict: 1/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P1 pick a card (main) → **pick_center**  search: root value -0.10; pick_right(opp deck) n=56 q=-0.18, pick_left(own deck) n=40 q=-0.22, pick_center n=204 q=-0.06

## Turn 27 — P0 to move

Table: P0 deck[25]: shield_h1 | P1 deck[20]: gear | center[36]: shield_h1 (known to [0])  
P0 Rhodes stage 3/5  score 18  shields 3  mil 0  cat yes  tokens ['Urbanism']
       cards: civ2catx2, compassx1, shieldx1, shield_h1x1  
P1 Ephesus stage 3/5  score 21  shields 2  mil 2  cat no  tokens []
       cards: stonex1, civ3x1, civ2catx1, gearx1, shieldx2  
Conflict: 1/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P0 pick a card (main) → **pick_right(opp deck)**  search: root value +0.21; pick_left(own deck) n=41 q=+0.11, pick_right(opp deck) n=210 q=+0.24, pick_center n=49 q=+0.16

## Turn 28 — P1 to move

Table: P0 deck[25]: shield_h1 | P1 deck[19]: civ2cat | center[36]: ?  
P0 Rhodes stage 3/5  score 18  shields 3  mil 0  cat yes  tokens ['Urbanism']
       cards: civ2catx2, gearx1, compassx1, shieldx1, shield_h1x1  
P1 Ephesus stage 3/5  score 21  shields 2  mil 2  cat no  tokens []
       cards: stonex1, civ3x1, civ2catx1, gearx1, shieldx2  
Conflict: 1/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P1 pick a card (main) → **pick_left(own deck)**  search: root value -0.17; pick_right(opp deck) n=47 q=-0.20, pick_left(own deck) n=174 q=-0.17, pick_center n=79 q=-0.16

## Turn 29 — P0 to move

Table: P0 deck[25]: shield_h1 | P1 deck[18]: papyrus | center[36]: shield_h1 (known to [0])  
P0 Rhodes stage 3/5  score 16  shields 3  mil 0  cat no  tokens ['Urbanism']
       cards: civ2catx2, gearx1, compassx1, shieldx1, shield_h1x1  
P1 Ephesus stage 3/5  score 25  shields 2  mil 2  cat yes  tokens []
       cards: stonex1, civ3x1, civ2catx2, gearx1, shieldx2  
Conflict: 1/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P0 pick a card (main) → **pick_right(opp deck)**  search: root value +0.15; pick_left(own deck) n=10 q=-0.14, pick_right(opp deck) n=281 q=+0.17, pick_center n=9 q=-0.18

## Turn 30 — P1 to move

Table: P0 deck[25]: shield_h1 | P1 deck[17]: coin | center[36]: shield_h1 (known to [0, 1])  
P0 Rhodes stage 3/5  score 16  shields 3  mil 0  cat no  tokens ['Urbanism']
       cards: papyrusx1, civ2catx2, gearx1, compassx1, shieldx1, shield_h1x1  
P1 Ephesus stage 3/5  score 25  shields 2  mil 2  cat yes  tokens []
       cards: stonex1, civ3x1, civ2catx2, gearx1, shieldx2  
Conflict: 1/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P1 pick a card (main) → **pick_left(own deck)**  search: root value -0.06; pick_right(opp deck) n=1 q=-0.41, pick_left(own deck) n=296 q=-0.06, pick_center n=3 q=-0.32

## Turn 31 — P0 to move

Table: P0 deck[25]: shield_h1 | P1 deck[16]: glass | center[36]: shield_h1 (known to [0, 1])  
P0 Rhodes stage 3/5  score 16  shields 3  mil 0  cat no  tokens ['Urbanism']
       cards: papyrusx1, civ2catx2, gearx1, compassx1, shieldx1, shield_h1x1  
P1 Ephesus stage 3/5  score 25  shields 2  mil 2  cat yes  tokens []
       cards: stonex1, coinx1, civ3x1, civ2catx2, gearx1, shieldx2  
Conflict: 1/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P0 pick a card (main) → **pick_right(opp deck)**  search: root value +0.02; pick_left(own deck) n=35 q=-0.22, pick_right(opp deck) n=247 q=+0.08, pick_center n=18 q=-0.41

## Turn 32 — P1 to move

Table: P0 deck[25]: shield_h1 | P1 deck[15]: clay | center[36]: shield_h1 (known to [0, 1])  
P0 Rhodes stage 3/5  score 16  shields 3  mil 0  cat no  tokens ['Urbanism']
       cards: papyrusx1, glassx1, civ2catx2, gearx1, compassx1, shieldx1, shield_h1x1  
P1 Ephesus stage 3/5  score 25  shields 2  mil 2  cat yes  tokens []
       cards: stonex1, coinx1, civ3x1, civ2catx2, gearx1, shieldx2  
Conflict: 1/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P1 pick a card (main) → **pick_left(own deck)**  search: root value -0.05; pick_right(opp deck) n=18 q=-0.22, pick_left(own deck) n=253 q=-0.02, pick_center n=29 q=-0.18

## Turn 33 — P0 to move

Table: P0 deck[25]: shield_h1 | P1 deck[14]: shield_h1 | center[36]: shield_h1 (known to [0, 1])  
P0 Rhodes stage 3/5  score 16  shields 3  mil 0  cat no  tokens ['Urbanism']
       cards: papyrusx1, glassx1, civ2catx2, gearx1, compassx1, shieldx1, shield_h1x1  
P1 Ephesus stage 3/5  score 25  shields 2  mil 2  cat yes  tokens []
       cards: stonex1, clayx1, coinx1, civ3x1, civ2catx2, gearx1, shieldx2  
Conflict: 1/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P0 pick a card (main) → **pick_right(opp deck)**  search: root value -0.12; pick_left(own deck) n=64 q=-0.17, pick_right(opp deck) n=197 q=-0.07, pick_center n=39 q=-0.25

## Turn 34 — P1 to move

Table: P0 deck[25]: shield_h1 | P1 deck[13]: glass | center[36]: shield_h1 (known to [0, 1])  
P0 Rhodes stage 3/5  score 16  shields 4  mil 0  cat no  tokens ['Urbanism']
       cards: papyrusx1, glassx1, civ2catx2, gearx1, compassx1, shieldx1, shield_h1x2  
P1 Ephesus stage 3/5  score 25  shields 2  mil 2  cat yes  tokens []
       cards: stonex1, clayx1, coinx1, civ3x1, civ2catx2, gearx1, shieldx2  
Conflict: 2/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P1 pick a card (main) → **pick_left(own deck)**  search: root value +0.19; pick_right(opp deck) n=40 q=+0.06, pick_left(own deck) n=143 q=+0.24, pick_center n=117 q=+0.18

## Turn 35 — P0 to move

Table: P0 deck[25]: shield_h1 | P1 deck[12]: shield | center[36]: shield_h1 (known to [0, 1])  
P0 Rhodes stage 3/5  score 16  shields 4  mil 0  cat no  tokens ['Urbanism']
       cards: papyrusx1, glassx1, civ2catx2, gearx1, compassx1, shieldx1, shield_h1x2  
P1 Ephesus stage 3/5  score 25  shields 2  mil 2  cat yes  tokens []
       cards: stonex1, clayx1, glassx1, coinx1, civ3x1, civ2catx2, gearx1, shieldx2  
Conflict: 2/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P0 pick a card (main) → **pick_right(opp deck)**  search: root value -0.41; pick_left(own deck) n=15 q=-0.59, pick_right(opp deck) n=272 q=-0.39, pick_center n=13 q=-0.69

## Turn 36 — P1 to move

Table: P0 deck[25]: shield_h1 | P1 deck[11]: tablet | center[36]: shield_h1 (known to [0, 1])  
P0 Rhodes stage 3/5  score 16  shields 5  mil 0  cat no  tokens ['Urbanism']
       cards: papyrusx1, glassx1, civ2catx2, gearx1, compassx1, shieldx2, shield_h1x2  
P1 Ephesus stage 3/5  score 25  shields 2  mil 2  cat yes  tokens []
       cards: stonex1, clayx1, glassx1, coinx1, civ3x1, civ2catx2, gearx1, shieldx2  
Conflict: 2/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P1 pick a card (main) → **pick_center**  search: root value +0.46; pick_right(opp deck) n=17 q=+0.13, pick_left(own deck) n=46 q=+0.25, pick_center n=237 q=+0.53

## Turn 37 — P0 to move

Table: P0 deck[25]: shield_h1 | P1 deck[11]: tablet | center[35]: ?  
P0 Rhodes stage 3/5  score 19  shields 3  mil 1  cat no  tokens ['Urbanism']
       cards: papyrusx1, glassx1, civ2catx2, gearx1, compassx1, shieldx2  
P1 Ephesus stage 3/5  score 25  shields 2  mil 2  cat yes  tokens []
       cards: stonex1, clayx1, glassx1, coinx1, civ3x1, civ2catx2, gearx1, shieldx2  
Conflict: 0/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P0 pick a card (main) → **pick_center**  search: root value -0.52; pick_left(own deck) n=15 q=-0.77, pick_right(opp deck) n=26 q=-0.70, pick_center n=259 q=-0.49

## Turn 38 — P1 to move

Table: P0 deck[25]: shield_h1 | P1 deck[11]: tablet | center[34]: ?  
P0 Rhodes stage 3/5  score 23  shields 3  mil 1  cat yes  tokens ['Urbanism']
       cards: papyrusx1, glassx1, civ2catx3, gearx1, compassx1, shieldx2  
P1 Ephesus stage 3/5  score 23  shields 2  mil 2  cat no  tokens []
       cards: stonex1, clayx1, glassx1, coinx1, civ3x1, civ2catx2, gearx1, shieldx2  
Conflict: 0/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P1 pick a card (main) → **pick_center**  search: root value +0.39; pick_right(opp deck) n=9 q=+0.22, pick_left(own deck) n=9 q=+0.23, pick_center n=282 q=+0.40

## Turn 39 — P0 to move

Table: P0 deck[25]: shield_h1 | P1 deck[11]: tablet | center[33]: civ3 (known to [0])  
P0 Rhodes stage 3/5  score 23  shields 3  mil 1  cat yes  tokens ['Urbanism']
       cards: papyrusx1, glassx1, civ2catx3, gearx1, compassx1, shieldx2  
P1 Ephesus stage 3/5  score 23  shields 3  mil 2  cat no  tokens []
       cards: stonex1, clayx1, glassx1, coinx1, civ3x1, civ2catx2, gearx1, shieldx2, shield_h2x1  
Conflict: 2/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P0 pick a card (main) → **pick_center**  search: root value -0.35; pick_left(own deck) n=104 q=-0.33, pick_right(opp deck) n=23 q=-0.53, pick_center n=173 q=-0.34

## Turn 40 — P1 to move

Table: P0 deck[25]: shield_h1 | P1 deck[11]: tablet | center[32]: ?  
P0 Rhodes stage 3/5  score 26  shields 3  mil 1  cat yes  tokens ['Urbanism']
       cards: papyrusx1, glassx1, civ3x1, civ2catx3, gearx1, compassx1, shieldx2  
P1 Ephesus stage 3/5  score 23  shields 3  mil 2  cat no  tokens []
       cards: stonex1, clayx1, glassx1, coinx1, civ3x1, civ2catx2, gearx1, shieldx2, shield_h2x1  
Conflict: 2/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P1 pick a card (main) → **pick_center**  search: root value +0.33; pick_right(opp deck) n=12 q=+0.07, pick_left(own deck) n=5 q=+0.07, pick_center n=283 q=+0.34

## Turn 41 — P0 to move

Table: P0 deck[25]: shield_h1 | P1 deck[11]: tablet | center[31]: civ3 (known to [0])  
P0 Rhodes stage 3/5  score 26  shields 3  mil 1  cat yes  tokens ['Urbanism']
       cards: papyrusx1, glassx1, civ3x1, civ2catx3, gearx1, compassx1, shieldx2  
P1 Ephesus stage 3/5  score 26  shields 3  mil 2  cat no  tokens []
       cards: stonex1, clayx1, glassx1, coinx1, civ3x2, civ2catx2, gearx1, shieldx2, shield_h2x1  
Conflict: 2/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P0 pick a card (main) → **pick_center**  search: root value -0.36; pick_left(own deck) n=74 q=-0.35, pick_right(opp deck) n=28 q=-0.50, pick_center n=198 q=-0.35

## Turn 42 — P1 to move

Table: P0 deck[25]: shield_h1 | P1 deck[11]: tablet | center[30]: ?  
P0 Rhodes stage 3/5  score 29  shields 3  mil 1  cat yes  tokens ['Urbanism']
       cards: papyrusx1, glassx1, civ3x2, civ2catx3, gearx1, compassx1, shieldx2  
P1 Ephesus stage 3/5  score 26  shields 3  mil 2  cat no  tokens []
       cards: stonex1, clayx1, glassx1, coinx1, civ3x2, civ2catx2, gearx1, shieldx2, shield_h2x1  
Conflict: 2/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P1 pick a card (main) → **pick_center**  search: root value +0.32; pick_right(opp deck) n=10 q=+0.01, pick_left(own deck) n=6 q=+0.14, pick_center n=284 q=+0.33

## Turn 43 — P0 to move

Table: P0 deck[25]: shield_h1 | P1 deck[11]: tablet | center[29]: shield_h1 (known to [0])  
P0 Rhodes stage 3/5  score 29  shields 3  mil 1  cat yes  tokens ['Urbanism']
       cards: papyrusx1, glassx1, civ3x2, civ2catx3, gearx1, compassx1, shieldx2  
P1 Ephesus stage 4/5  score 32  shields 3  mil 2  cat no  tokens []
       cards: clayx1, glassx1, civ3x2, civ2catx2, gearx1, shieldx2, shield_h2x1  
Conflict: 2/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P0 pick a card (main) → **pick_left(own deck)**  search: root value -0.49; pick_left(own deck) n=131 q=-0.47, pick_right(opp deck) n=54 q=-0.56, pick_center n=115 q=-0.47

## Turn 44 — P1 to move

Table: P0 deck[24]: tablet | P1 deck[11]: tablet | center[29]: ?  
P0 Rhodes stage 3/5  score 32  shields 3  mil 2  cat yes  tokens ['Urbanism']
       cards: papyrusx1, glassx1, civ3x2, civ2catx3, gearx1, compassx1, shieldx2  
P1 Ephesus stage 4/5  score 32  shields 2  mil 2  cat no  tokens []
       cards: clayx1, glassx1, civ3x2, civ2catx2, gearx1, shieldx2  
Conflict: 0/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P1 pick a card (main) → **pick_center**  search: root value +0.50; pick_right(opp deck) n=60 q=+0.48, pick_left(own deck) n=46 q=+0.48, pick_center n=194 q=+0.52

## Turn 45 — P0 to move

Table: P0 deck[24]: tablet | P1 deck[11]: tablet | center[28]: shield_h2 (known to [0])  
P0 Rhodes stage 3/5  score 32  shields 3  mil 2  cat yes  tokens ['Urbanism']
       cards: papyrusx1, glassx1, civ3x2, civ2catx3, gearx1, compassx1, shieldx2  
P1 Ephesus stage 4/5  score 32  shields 3  mil 2  cat no  tokens []
       cards: clayx1, glassx1, civ3x2, civ2catx2, gearx1, shieldx2, shield_h1x1  
Conflict: 1/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P0 pick a card (main) → **pick_center**  search: root value -0.28; pick_left(own deck) n=117 q=-0.24, pick_right(opp deck) n=48 q=-0.39, pick_center n=135 q=-0.26

## Turn 46 — P1 to move

Table: P0 deck[24]: tablet | P1 deck[11]: tablet | center[27]: ?  
P0 Rhodes stage 3/5  score 35  shields 3  mil 3  cat yes  tokens ['Urbanism']
       cards: papyrusx1, glassx1, civ3x2, civ2catx3, gearx1, compassx1, shieldx2  
P1 Ephesus stage 4/5  score 32  shields 2  mil 2  cat no  tokens []
       cards: clayx1, glassx1, civ3x2, civ2catx2, gearx1, shieldx2  
Conflict: 0/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P1 pick a card (main) → **pick_center**  search: root value +0.26; pick_right(opp deck) n=67 q=+0.26, pick_left(own deck) n=76 q=+0.30, pick_center n=157 q=+0.24

## Turn 47 — P0 to move

Table: P0 deck[24]: tablet | P1 deck[11]: tablet | center[26]: wood (known to [0])  
P0 Rhodes stage 3/5  score 35  shields 3  mil 3  cat yes  tokens ['Urbanism']
       cards: papyrusx1, glassx1, civ3x2, civ2catx3, gearx1, compassx1, shieldx2  
P1 Ephesus stage 4/5  score 32  shields 2  mil 2  cat no  tokens []
       cards: clayx1, glassx2, civ3x2, civ2catx2, gearx1, shieldx2  
Conflict: 0/3, face-up tokens ['Culture', 'Propaganda', 'Education']

- P0 pick a card (main) → **pick_center**  search: root value +0.14; pick_left(own deck) n=3 q=-0.22, pick_right(opp deck) n=7 q=-0.05, pick_center n=290 q=+0.15
- P0 pick a card (token:2, optional) → **pick_left(own deck)**  search: root value +0.15; pick_left(own deck) n=230 q=+0.20, pick_right(opp deck) n=168 q=+0.16, pick_center n=180 q=+0.09, skip n=6 q=-0.31
- P0 choose a Progress token (science); face-up: ['Culture', 'Propaganda', 'Education'] → **token_Culture**  search: root value +0.14; token_Propaganda n=1 q=-0.31, token_Education n=117 q=+0.17, token_Culture n=173 q=+0.14, token_blind n=19 q=+0.07

## Turn 48 — P1 to move

Table: P0 deck[23]: compass | P1 deck[11]: tablet | center[25]: ?  
P0 Rhodes stage 3/5  score 39  shields 3  mil 3  cat yes  tokens ['Urbanism', 'Culture']
       cards: woodx1, papyrusx1, glassx1, civ3x2, civ2catx3, shieldx2  
P1 Ephesus stage 4/5  score 32  shields 2  mil 2  cat no  tokens []
       cards: clayx1, glassx2, civ3x2, civ2catx2, gearx1, shieldx2  
Conflict: 0/3, face-up tokens ['Propaganda', 'Education', 'Strategy']

- P1 pick a card (main) → **pick_center**  search: root value -0.24; pick_right(opp deck) n=21 q=-0.43, pick_left(own deck) n=64 q=-0.23, pick_center n=215 q=-0.23

## Turn 49 — P0 to move

Table: P0 deck[23]: compass | P1 deck[11]: tablet | center[24]: coin (known to [0])  
P0 Rhodes stage 3/5  score 39  shields 3  mil 3  cat yes  tokens ['Urbanism', 'Culture']
       cards: woodx1, papyrusx1, glassx1, civ3x2, civ2catx3, shieldx2  
P1 Ephesus stage 4/5  score 32  shields 3  mil 2  cat no  tokens []
       cards: clayx1, glassx2, civ3x2, civ2catx2, gearx1, shieldx2, shield_h1x1  
Conflict: 1/3, face-up tokens ['Propaganda', 'Education', 'Strategy']

- P0 pick a card (main) → **pick_center**  search: root value +0.49; pick_left(own deck) n=3 q=+0.42, pick_right(opp deck) n=2 q=+0.28, pick_center n=295 q=+0.49

## Turn 50 — P1 to move

Table: P0 deck[23]: compass | P1 deck[11]: tablet | center[23]: ?  
P0 Rhodes stage 3/5  score 39  shields 3  mil 3  cat yes  tokens ['Urbanism', 'Culture']
       cards: woodx1, papyrusx1, glassx1, coinx1, civ3x2, civ2catx3, shieldx2  
P1 Ephesus stage 4/5  score 32  shields 3  mil 2  cat no  tokens []
       cards: clayx1, glassx2, civ3x2, civ2catx2, gearx1, shieldx2, shield_h1x1  
Conflict: 1/3, face-up tokens ['Propaganda', 'Education', 'Strategy']

- P1 pick a card (main) → **pick_center**  search: root value -0.48; pick_right(opp deck) n=20 q=-0.66, pick_left(own deck) n=32 q=-0.52, pick_center n=248 q=-0.46

## Turn 51 — P0 to move

Table: P0 deck[23]: compass | P1 deck[11]: tablet | center[22]: coin (known to [0])  
P0 Rhodes stage 3/5  score 39  shields 3  mil 3  cat yes  tokens ['Urbanism', 'Culture']
       cards: woodx1, papyrusx1, glassx1, coinx1, civ3x2, civ2catx3, shieldx2  
P1 Ephesus stage 4/5  score 32  shields 3  mil 2  cat no  tokens []
       cards: clayx2, glassx2, civ3x2, civ2catx2, gearx1, shieldx2, shield_h1x1  
Conflict: 1/3, face-up tokens ['Propaganda', 'Education', 'Strategy']

- P0 pick a card (main) → **pick_center**  search: root value +0.89; pick_left(own deck) n=2 q=+0.53, pick_right(opp deck) n=3 q=+0.63, pick_center n=295 q=+0.89
- P0 pay for stage 4 (3 identical), paid so far [] → **pay_wood**  search: root value +0.89; pay_wood n=215 q=+0.90, pay_papyrus n=197 q=+0.90, pay_glass n=177 q=+0.89

## Turn 52 — P1 to move

Table: P0 deck[23]: compass | P1 deck[11]: tablet | center[21]: ?  
P0 Rhodes stage 4/5  score 45  shields 4  mil 3  cat yes  tokens ['Urbanism', 'Culture']
       cards: papyrusx1, glassx1, civ3x2, civ2catx3, shieldx2  
P1 Ephesus stage 4/5  score 32  shields 3  mil 2  cat no  tokens []
       cards: clayx2, glassx2, civ3x2, civ2catx2, gearx1, shieldx2, shield_h1x1  
Conflict: 1/3, face-up tokens ['Propaganda', 'Education', 'Strategy']

- P1 pick a card (main) → **pick_center**  search: root value -0.91; pick_right(opp deck) n=41 q=-0.95, pick_left(own deck) n=37 q=-0.93, pick_center n=222 q=-0.89

## Turn 53 — P0 to move

Table: P0 deck[23]: compass | P1 deck[11]: tablet | center[20]: clay (known to [0])  
P0 Rhodes stage 4/5  score 45  shields 4  mil 3  cat yes  tokens ['Urbanism', 'Culture']
       cards: papyrusx1, glassx1, civ3x2, civ2catx3, shieldx2  
P1 Ephesus stage 4/5  score 35  shields 3  mil 2  cat no  tokens []
       cards: clayx2, glassx2, civ3x3, civ2catx2, gearx1, shieldx2, shield_h1x1  
Conflict: 1/3, face-up tokens ['Propaganda', 'Education', 'Strategy']

- P0 pick a card (main) → **pick_center**  search: root value +0.91; pick_left(own deck) n=3 q=+0.84, pick_right(opp deck) n=7 q=+0.90, pick_center n=290 q=+0.91
- P0 pick a card (token:2, optional) → **pick_center**  search: root value +0.92; pick_left(own deck) n=0, pick_right(opp deck) n=0, pick_center n=584 q=+0.92, skip n=0

## Turn 54 — P1 to move

Table: P0 deck[23]: compass | P1 deck[11]: tablet | center[18]: ?  
P0 Rhodes stage 4/5  score 47  shields 4  mil 3  cat yes  tokens ['Urbanism', 'Culture']
       cards: clayx1, papyrusx1, glassx1, civ3x2, civ2catx4, shieldx2  
P1 Ephesus stage 4/5  score 35  shields 3  mil 2  cat no  tokens []
       cards: clayx2, glassx2, civ3x3, civ2catx2, gearx1, shieldx2, shield_h1x1  
Conflict: 1/3, face-up tokens ['Propaganda', 'Education', 'Strategy']

- P1 pick a card (main) → **pick_center**  search: root value -0.93; pick_right(opp deck) n=29 q=-0.97, pick_left(own deck) n=26 q=-0.96, pick_center n=245 q=-0.92

## Turn 55 — P0 to move

Table: P0 deck[23]: compass | P1 deck[11]: tablet | center[17]: shield_h1 (known to [0])  
P0 Rhodes stage 4/5  score 47  shields 4  mil 3  cat yes  tokens ['Urbanism', 'Culture']
       cards: clayx1, papyrusx1, glassx1, civ3x2, civ2catx4, shieldx2  
P1 Ephesus stage 4/5  score 35  shields 3  mil 2  cat no  tokens []
       cards: clayx2, glassx2, civ3x3, civ2catx2, tabletx1, gearx1, shieldx2, shield_h1x1  
Conflict: 1/3, face-up tokens ['Propaganda', 'Education', 'Strategy']

- P0 pick a card (main) → **pick_right(opp deck)**  search: root value +0.92; pick_left(own deck) n=76 q=+0.93, pick_right(opp deck) n=144 q=+0.95, pick_center n=80 q=+0.86

## Turn 56 — P1 to move

Table: P0 deck[23]: compass | P1 deck[10]: civ2cat | center[17]: ?  
P0 Rhodes stage 4/5  score 47  shields 4  mil 3  cat yes  tokens ['Urbanism', 'Culture']
       cards: clayx1, papyrusx1, glassx1, civ3x2, civ2catx4, tabletx1, shieldx2  
P1 Ephesus stage 4/5  score 35  shields 3  mil 2  cat no  tokens []
       cards: clayx2, glassx2, civ3x3, civ2catx2, tabletx1, gearx1, shieldx2, shield_h1x1  
Conflict: 1/3, face-up tokens ['Propaganda', 'Education', 'Strategy']

- P1 pick a card (main) → **pick_center**  search: root value -0.92; pick_right(opp deck) n=18 q=-0.91, pick_left(own deck) n=102 q=-0.93, pick_center n=180 q=-0.91

## Turn 57 — P0 to move

Table: P0 deck[23]: compass | P1 deck[10]: civ2cat | center[16]: coin (known to [0])  
P0 Rhodes stage 4/5  score 47  shields 4  mil 3  cat yes  tokens ['Urbanism', 'Culture']
       cards: clayx1, papyrusx1, glassx1, civ3x2, civ2catx4, tabletx1, shieldx2  
P1 Ephesus stage 4/5  score 35  shields 4  mil 2  cat no  tokens []
       cards: clayx2, glassx2, civ3x3, civ2catx2, tabletx1, gearx1, shieldx2, shield_h1x2  
Conflict: 2/3, face-up tokens ['Propaganda', 'Education', 'Strategy']

- P0 pick a card (main) → **pick_center**  search: root value +1.00; pick_left(own deck) n=0, pick_right(opp deck) n=0, pick_center n=300 q=+1.00

## Final position

P0 Rhodes stage 5/5  score 55  shields 4  mil 3  cat yes  tokens ['Urbanism', 'Culture']
       cards: civ3x2, civ2catx4, tabletx1, shieldx2  
P1 Ephesus stage 4/5  score 35  shields 4  mil 2  cat no  tokens []
       cards: clayx2, glassx2, civ3x3, civ2catx2, tabletx1, gearx1, shieldx2, shield_h1x2  

**Result: P0** — scores 55–35 after 57 turns, 63 decisions (22s).
