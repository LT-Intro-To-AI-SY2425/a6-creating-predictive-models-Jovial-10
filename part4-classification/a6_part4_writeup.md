# Part 4 - Classification Writeup

After completing `a6_part4.py` answer the following questions

## Questions to answer

1. Comment out the StandardScaler and re-run your test. How accurate is the model? Why is that?

After commenting out the StandarScaler, my models accuracy was 0.66. Since the data was not scaled, the accuracy was a lot lower and some data may have been skewed. 

2. How accurate is the model with the StandardScaler? Is this model accurate enough for the given use case? Explain.

It is about 0.85-0.9. This is model is a lot more accurate than the model without StandarScaler and accurate enough for the given case.

3. Looking at the predicted and actual results, how did the model do? Was there a pattern to the inputs that the model was incorrect about?

Most of the predicted and actual results were the same, with one or two predicitons being incorrect. Yes, most of the incorrect predictions were 0 while the actual species was 1.


4. Would a 34 year old Female who makes 56000 a year buy an SUV according to the model? Remember to scale the data before running it through the model.

No, the 34 year old female who makes 56,000 a year would not be likely to buy an SUV.
