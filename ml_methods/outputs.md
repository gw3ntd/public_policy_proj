# Output Log
### Here, I am keeping track of the current outputs of my functions in the 'ml_attempt.py' file. 

## get_importance(df, model, color='purple')
I originally made this function to get the importance scores for each feature and compare the 
importance scores accross models. However, when doing this, I was getting a very low cv
score for each model. 

I tried the following:
'get_importance(df, DecisionTreeRegressor(random_state=42))'
'get_importance(df, RandomForestRegressor(random_state=42), color='teal')'
'get_importance(df, GradientBoostingRegressor(random_state=42), color='pink')'

### Output:
Mean r^2 score accross 5 folds for DecisionTreeRegressor(random_state=42): 0.236
All fold scores for DecisionTreeRegressor(random_state=42): [ 0.23092503  0.77619647  0.7652932   0.29895134 -0.89273118]

Fams Below Pov %, Score: 0.07457
Ppl (<150% Of Pov) %, Score: 0.00374
Ppl (Below Poverty) %, Score: 0.04395
% At Least Bachelor's Degree, Score: 0.47181
Median Fam Income (Scaled), Score: 0.02827
Median Household Income (Scaled), Score: 0.16602
Vast Majority Income (Scaled), Score: 0.00103
Language Isolation %, Score: 0.21060


Mean r^2 score accross 5 folds for RandomForestRegressor(random_state=42): 0.491
All fold scores for RandomForestRegressor(random_state=42): [ 0.4201826   0.64573927  0.7237883   0.69102035 -0.02674616]

Fams Below Pov %, Score: 0.05428
Ppl (<150% Of Pov) %, Score: 0.03473
Ppl (Below Poverty) %, Score: 0.03383
% At Least Bachelor's Degree, Score: 0.30526
Median Fam Income (Scaled), Score: 0.18209
Median Household Income (Scaled), Score: 0.11751
Vast Majority Income (Scaled), Score: 0.11447
Language Isolation %, Score: 0.15783

Mean r^2 score accross 5 folds for GradientBoostingRegressor(random_state=42): 0.432
All fold scores for GradientBoostingRegressor(random_state=42): [ 0.2980692   0.6816932   0.73679643  0.70031228 -0.25924952]

Fams Below Pov %, Score: 0.03618
Ppl (<150% Of Pov) %, Score: 0.03724
Ppl (Below Poverty) %, Score: 0.02374
% At Least Bachelor's Degree, Score: 0.39926
Median Fam Income (Scaled), Score: 0.10813
Median Household Income (Scaled), Score: 0.13981
Vast Majority Income (Scaled), Score: 0.06105
Language Isolation %, Score: 0.19459

## all_iterations(df, combos)
Then, I tried finding the combonations of 3 and 4 features that would have a 5-fold cv score greater than
0.5. This is what I wanted the cv score to be originally, but no model applied to this. This function just 
returned a list of indicies to which this applied. I also printed out which combonations had the desired
score. 

'L1 = all_iterations(df, combos)'

### Output
8: ['Fams Below Pov %', 'Ppl (<150% Of Pov) %', "% At Least Bachelor's Degree", 'Language Isolation %']
Mean r^2 score accross 5 folds for RandomForestRegressor(): 0.522
All fold scores for RandomForestRegressor(): [0.45380844 0.55189209 0.68377805 0.81658613 0.10444955]

18: ['Fams Below Pov %', 'Ppl (Below Poverty) %', "% At Least Bachelor's Degree", 'Language Isolation %']
Mean r^2 score accross 5 folds for RandomForestRegressor(): 0.517
All fold scores for RandomForestRegressor(): [0.41778408 0.54202598 0.64617244 0.8563232  0.12264052]

29: ['Fams Below Pov %', "% At Least Bachelor's Degree", 'Median Household Income (Scaled)', 'Language Isolation %']
Mean r^2 score accross 5 folds for RandomForestRegressor(): 0.518
All fold scores for RandomForestRegressor(): [0.50549955 0.53792553 0.6362863  0.6291766  0.28075132]

30: ['Fams Below Pov %', "% At Least Bachelor's Degree", 'Vast Majority Income (Scaled)', 'Language Isolation %']
Mean r^2 score accross 5 folds for RandomForestRegressor(): 0.532
All fold scores for RandomForestRegressor(): [ 0.47673669  0.616639    0.73217154  0.90751257 -0.07313962]

50: ['Ppl (<150% Of Pov) %', "% At Least Bachelor's Degree", 'Vast Majority Income (Scaled)', 'Language Isolation %']
Mean r^2 score accross 5 folds for RandomForestRegressor(): 0.522
All fold scores for RandomForestRegressor(): [ 0.38042699  0.67644018  0.73737917  0.89219223 -0.0761083 ]

60: ['Ppl (Below Poverty) %', "% At Least Bachelor's Degree", 'Vast Majority Income (Scaled)', 'Language Isolation %']
Mean r^2 score accross 5 folds for RandomForestRegressor(): 0.527
All fold scores for RandomForestRegressor(): [ 0.39634329  0.67000959  0.70272991  0.91395034 -0.04670631]

## div_importance(df, combos)
I found the new feature importance scores to the columns that applied. Interestingly, % Above Bachelor's Ed
appeared in every single one. 

### Output:
18
Fams Below Pov %, Score: 0.13403
Ppl (Below Poverty) %, Score: 0.14278
% At Least Bachelor's Degree, Score: 0.48726
Language Isolation %, Score: 0.23592

29
Fams Below Pov %, Score: 0.16724
% At Least Bachelor's Degree, Score: 0.38614
Median Household Income (Scaled), Score: 0.27965
Language Isolation %, Score: 0.16697

38
Ppl (<150% Of Pov) %, Score: 0.16140
Ppl (Below Poverty) %, Score: 0.11628
% At Least Bachelor's Degree, Score: 0.51393
Language Isolation %, Score: 0.20839

50
Ppl (<150% Of Pov) %, Score: 0.11478
% At Least Bachelor's Degree, Score: 0.35759
Vast Majority Income (Scaled), Score: 0.32066
Language Isolation %, Score: 0.20698

60
Ppl (Below Poverty) %, Score: 0.10826
% At Least Bachelor's Degree, Score: 0.39986
Vast Majority Income (Scaled), Score: 0.30149
Language Isolation %, Score: 0.19039

***The following is when I tried it with 3 variables:***

14
Fams Below Pov %, Score: 0.20983
% At Least Bachelor's Degree, Score: 0.56010
Language Isolation %, Score: 0.23006

29
Ppl (<150% Of Pov) %, Score: 0.26481
% At Least Bachelor's Degree, Score: 0.51004
Language Isolation %, Score: 0.22515

39
Ppl (Below Poverty) %, Score: 0.22573
% At Least Bachelor's Degree, Score: 0.53830
Language Isolation %, Score: 0.23597

