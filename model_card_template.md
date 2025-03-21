# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model used here relies on SKLearn's RandomForestClassifier, called as RFC for simplicity, with the goal of predicting whether a person's income is above or below $50,000. Training data for the model draws from the US Census data contained within the project in the data folder. Categorization of data is as follows:

## Intended Use

This model was created for a class project. It is intended to use relevant demographic data to determine a person's income level, "<=50k" or ">50k".

## Training Data

The training data was provided by the project and can be found in the data folder labeled as census.csv. This is a subset of data from the 1994 Census. Original data and further details can be found here: https://archive.ics.uci.edu/dataset/20/census+income. Below I have provided a brief description of the fields and their relevance.

### age:
- Integer value representing age of the person.

### workclass:
- Categorical classification representing a person's type of employement.

### fnlgt:
- Integer value. Unclear what this field represents in the data. Likely some sort of unique record identifier.

### education:
- Categorical classification representing a person's education level.

### education-num
- Integer value representing a person's education level. Correlates to education.

### marital-status
- Categorical classification representing a person's marital status.

### occupation
- Categorical classification representing a person's type of occupation.

### relationship:
- Categorical classification representing a person's relationship status.

### race:
- Categorical classification representing a person's ethnicity.

### sex:
- Categorical classification representing a person's male/female gender.

### capital-gain:
- Integer value representing a person's capital gains.

### capital-loss:
- Integer value representing a person's capital losses.

### hours-per-week:
- Integer value showing the number of hours a person worked per week.

### native-country:
- Categorical classification representing a person's country of origin.

### salary:
- Categorical classification representing a person's income as "<=50k" or ">50k".

## Evaluation Data

The data was split into a training dataset and a test dataset. The test dataset split was 20% of the overall dataset.

## Metrics
_Please include the metrics used and your model's performance on those metrics._

## Ethical Considerations

There are several potential considerations in this dataset. Some distinctions are noted below:

### Gender:
- Males represented 66.9% of the dataset surveyed. Given the wage gap between males and females in the same roles, it is safe to assume this skewed the data higher than a truly even sample would have provided.

### Race:
- White people represented 85.4% of the surveyed population. While this may be representative of the population in certain areas of the US, there are many regions where minority groups cumulatively make up a larger portion of the population. Historically minority gorups in the same role as white people make less money. Therefore, if this proportion is not representative of the target population, it would likely skew the results.

## Caveats and Recommendations

### Caveat:
- This data is clearly sampled from a predominantly white population and should not be used to represent a population where that does not hold true. 

### Recommendations:
If the full census dataset were to be used, instead of this subset, it would include data such as locality. This could be used to allow for more targeted sampling and modeling of the data, ultimately leading to a more accurate prediciton model that would align better with the targeted population.
