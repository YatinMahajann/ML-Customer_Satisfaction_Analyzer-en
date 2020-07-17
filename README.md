# ML-Customer_Satisfaction_Analyzer-en
To predict from the sentence whether the user is Satisfied, Unsatisfied or Neutral

Training of the Model
--------------------------------------------------
---------------------------------------------------
1. Initial model was completely trained using State of the Art BERT Model. But due to trade off between speed and performance, we have to go with other alternate option.

2. The current model uses the embedding of Distill Bert.

3. Rather than using Distill BERT's classifier, the embeddings are now passed to Random Foerst Classifier which significantly increased the response time of the request.

4. At the time of testing, on the local system using FLASK API the response time was  between 65 - 85 ms per request.

5. The above file was passed to Akshay to make the necessary changes and handle exceptions to further integrate with the chatbot platform.
