Homework 3
================
Erik Carlson
10/5/2020

## Knn Borough Prediction

This homework was done with my study group of Emmanuel Mendez, Emily
Vasquez, and Joe Correa.

We started by using the knn method to predict the Borough of the
residents, later this method was used to predict neighborhoods. First
the acs data was loaded into R and the different boroughs were selected
as factors for later comparison. Since we are only considerned with
neighboorhoods and boroughs in New York City, the data was subsetted to
include only those in New York City and those in an age range between 20
and 60.

``` r
load("acs2017_ny_data.RData")
PUMA_lev<-read.csv("PUMA_levels.csv")
dat_NYC <- subset(acs2017_ny, (acs2017_ny$in_NYC == 1)&(acs2017_ny$AGE > 20) & (acs2017_ny$AGE < 66))
attach(dat_NYC)
borough_f <- factor((in_Bronx + 2*in_Manhattan 
                     + 3*in_StatenI + 4*in_Brooklyn + 5*in_Queens), 
                    levels=c(1,2,3,4,5),
                    labels = c("Bronx","Manhattan","Staten Island","Brooklyn","Queens"))
```

In order to classify by borough with a high accuracy, selection of
variables is critical. By using housing cost and total income, a correct
rate of about 37% was found. Given that the sample proportion of
Brooklyn is 35.3%, is it only marginally better than guessing that every
resident is in Brooklyn. Therefore other variables were tested and
selected.

It was thought that differences in the cost of gas and other housing
costs such as fuel, waer, electricity along with diferences in family
size would differ across the different boroughs. Therefore given the
correct variables, the sum of them would compound these differences,
allowing for better guessing. Variables and the sum of different
variables were selected to maximize the correct guess rate.

Household income was found to be a better indicator than total income,
so it was selected instead.

``` r
norm_varb <- function(X_in) {
  (max(X_in, na.rm = TRUE) - X_in)/( max(X_in, na.rm = TRUE) - min(X_in, na.rm = TRUE) )
}
  
add_t<- COSTGAS + COSTFUEL + COSTWATR + COSTELEC + FAMSIZE + FUELHEAT 
norm_1 <- norm_varb(add_t)
norm_2 <- norm_varb(HHINCOME)
```

Following this the algrorithm was trained with 80% of the data, and
tested with 20%.

``` r
#data_use_prelim <- data.frame(norm_1,norm_2) 
# Above Code commented out to solve knitting error. but result was loaded from workspace
good_obs_data_use <- complete.cases(data_use_prelim,borough_f)
dat_use <- subset(data_use_prelim,good_obs_data_use)
y_use <- subset(borough_f,good_obs_data_use)

set.seed(12345)
NN_obs <- sum(good_obs_data_use == 1)
select1 <- (runif(NN_obs) < 0.8)
train_data <- subset(dat_use,select1)
test_data <- subset(dat_use,(!select1))
cl_data <- y_use[select1]
true_data <- y_use[!select1]
```

And the knn algorithm was run and compared with the true data in order
to find the correct rate.

``` r
summary(cl_data)
```

    ##         Bronx     Manhattan Staten Island      Brooklyn        Queens 
    ##          4613          4896          1839         12073         10710

``` r
prop.table(summary(cl_data))
```

    ##         Bronx     Manhattan Staten Island      Brooklyn        Queens 
    ##    0.13515572    0.14344731    0.05388064    0.35372535    0.31379098

``` r
summary(train_data)
```

    ##      norm_1           norm_2      
    ##  Min.   :0.0000   Min.   :0.0000  
    ##  1st Qu.:0.2232   1st Qu.:0.9229  
    ##  Median :0.4452   Median :0.9537  
    ##  Mean   :0.3930   Mean   :0.9374  
    ##  3rd Qu.:0.5675   3rd Qu.:0.9755  
    ##  Max.   :0.9779   Max.   :1.0000

``` r
require(class)
for (indx in seq(1, 9, by= 2)) {
  pred_borough <- knn(train_data, test_data, cl_data, k = indx, l = 0, prob = FALSE, use.all = TRUE)
  num_correct_labels <- sum(pred_borough == true_data)
  correct_rate <- num_correct_labels/length(true_data)
  print(c(indx,correct_rate))
}
```

    ## [1] 1.000000 0.798117
    ## [1] 3.0000000 0.5191276
    ## [1] 5.0000000 0.5039924
    ## [1] 7.0000000 0.4931474
    ## [1] 9.000000 0.487427

The correct rate was found for to be 79.8% for k nearest neighbor(where
k=1) and for k\>1 it is less, about 50% overall. K=1 had the highest
accuracy, much higher than the original 37% or the 35% Brooklyn guess.

Normally k=1 may be inaccurate due to overfitting to the test data, this
overfitting leading to inaccuracies when using a different dataset.
Therefore higher k’s may bring greater accurracy if overfitting is an
issue. However, given that k=1 is the most accurate, it can be seen that
overfitting is not as severe an issue and that it is possible that there
are clear boundaries for the variables that k nearest neighbor analysis
is optimal for. In order to confirm that this is due to clear
boundaries, additional datasets need to be tested. K=1 may be the most
accurate, however a lower amount of neighbors make it more suceptable to
biases in the original sample.

The Borough proportion of the sample was graphed below in order to show
the makeup of the sample.

![](hw-3_Erik_Carlson_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

## Neighborhood Prediction

Following guessing the borough, neighborhood was confisidered. The
neighborhood is listed in the column “PUMA”, and is four numbers long.
The corresponding neighborhood for these codes is located in a
PUMA\_level csv file.

Given that the factor is now the neighborhood instead of borough, the
different neighborhood codes in PUMA was used instead of the boroughs.
Other possible variables were considered, however after testing these
same variables were effective in classifying by neighborhood.

``` r
borough_f <- factor(PUMA)

add_t<- COSTGAS + COSTFUEL + COSTWATR + COSTELEC + FAMSIZE + FUELHEAT 

norm_1 <- norm_varb(add_t)
norm_2 <- norm_varb(HHINCOME)
```

The data was used to train and test the algorithm using the methodology
that was used for boroughs.

``` r
#data_use_prelim <- data.frame(norm_1,norm_2)
# Above Code commented out to solve knitting error. but result was loaded from workspace
good_obs_data_use <- complete.cases(data_use_prelim,borough_f)
dat_use <- subset(data_use_prelim,good_obs_data_use)
y_use <- subset(borough_f,good_obs_data_use)

set.seed(12345)
NN_obs <- sum(good_obs_data_use == 1)
select1 <- (runif(NN_obs) < 0.8)
train_data <- subset(dat_use,select1)
test_data <- subset(dat_use,(!select1))
cl_data <- y_use[select1]
true_data <- y_use[!select1]
```

``` r
summary(cl_data)
```

    ## 3701 3702 3703 3704 3705 3706 3707 3708 3709 3710 3801 3802 3803 3804 3805 3806 
    ##  375  515  339  477  565  377  470  432  592  471  643  370  474  367  543  459 
    ## 3807 3808 3809 3810 3901 3902 3903 4001 4002 4003 4004 4005 4006 4007 4008 4009 
    ##  565  392  497  586  575  511  753  706  580  555  710  646  592  425  617 1137 
    ## 4010 4011 4012 4013 4014 4015 4016 4017 4018 4101 4102 4103 4104 4105 4106 4107 
    ##  599  390  652  756  660  696  858 1116  378  995  641 1118  631  970  600  515 
    ## 4108 4109 4110 4111 4112 4113 4114 
    ##  502  633 1029  885 1179  630  382

``` r
prop.table(summary(cl_data))
```

    ##       3701       3702       3703       3704       3705       3706       3707 
    ## 0.01098708 0.01508892 0.00993232 0.01397556 0.01655387 0.01104568 0.01377047 
    ##       3708       3709       3710       3801       3802       3803       3804 
    ## 0.01265712 0.01734494 0.01379977 0.01883918 0.01084058 0.01388767 0.01075269 
    ##       3805       3806       3807       3808       3809       3810       3901 
    ## 0.01590929 0.01344818 0.01655387 0.01148516 0.01456154 0.01716914 0.01684685 
    ##       3902       3903       4001       4002       4003       4004       4005 
    ## 0.01497173 0.02206206 0.02068501 0.01699335 0.01626088 0.02080220 0.01892708 
    ##       4006       4007       4008       4009       4010       4011       4012 
    ## 0.01734494 0.01245202 0.01807741 0.03331282 0.01755003 0.01142656 0.01910287 
    ##       4013       4014       4015       4016       4017       4018       4101 
    ## 0.02214995 0.01933726 0.02039202 0.02513844 0.03269755 0.01107498 0.02915238 
    ##       4102       4103       4104       4105       4106       4107       4108 
    ## 0.01878058 0.03275615 0.01848759 0.02841991 0.01757933 0.01508892 0.01470804 
    ##       4109       4110       4111       4112       4113       4114 
    ## 0.01854619 0.03014855 0.02592951 0.03454338 0.01845829 0.01119217

``` r
summary(train_data)
```

    ##      norm_1           norm_2      
    ##  Min.   :0.0000   Min.   :0.0000  
    ##  1st Qu.:0.2232   1st Qu.:0.9229  
    ##  Median :0.4452   Median :0.9537  
    ##  Mean   :0.3930   Mean   :0.9374  
    ##  3rd Qu.:0.5675   3rd Qu.:0.9755  
    ##  Max.   :0.9779   Max.   :1.0000

``` r
require(class)
for (indx in seq(1, 9, by= 2)) {
  pred_borough <- knn(train_data, test_data, cl_data, k = indx, l = 0, prob = FALSE, use.all = TRUE)
  num_correct_labels <- sum(pred_borough == true_data)
  correct_rate <- num_correct_labels/length(true_data)
  print(c(indx,correct_rate))
}
```

    ## [1] 1.0000000 0.6981289
    ## [1] 3.0000000 0.2829222
    ## [1] 5.0000000 0.2053391
    ## [1] 7.0000000 0.1749494
    ## [1] 9.0000000 0.1581456

Using this method, for k nearest neighbor(k=1) was 69.8% accurate while
k=3 was 28.3% accurate, where accuracy decreased as k increased. Similar
to the bourough test, the algoritm was most accurate when k was equal to
one. It was surprising that the same methodology only led to a 10%
decrease in accuracy for k=1 since there are many more possible
neighborhoods when compared with the 5 boroughs. Additional datasets
would be necessary in order to determine whether this is true for data
outside the training and testing data. However overall, neighborhoods
were guessed with a considerably larger probability, even with higher
k’s, compared to any possible random guess.

The Neighboorhood proportion of the sample was graphed below in order to
show the makeup of the sample.

![](hw-3_Erik_Carlson_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->
