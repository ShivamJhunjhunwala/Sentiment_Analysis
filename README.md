

# Sentiment Analysis Using Tensorflow

Sentiment analysis as a field has come a long way since it was first introduced as a task nearly 20 years ago. It has
widespread commercial applications in various domains like marketing, risk management, market research, and politics, to name a few. Given its saturation in specific subtasks — such as sentiment polarity classification and datasets, there is an underlying perception that this field has reached its maturity.

Here in this repository I have made a very simple and easy to understand sentiment analysis for Google Map reviews [@Hershey Theme Park](https://www.hersheypark.com/) using Tensorflow and Keras.


## Dataset Information

I have used my [@Google Map Reviews Scrapper](https://github.com/ShivamJhunjhunwala/GoogleMap_Data_Scraper) to scrap reviews for [@Hershey Theme Park](https://www.hersheypark.com/). the following will generate a csv file with Reviewer name, reviews, stars, etc.


## Installation 

```bash
pip install -r requirements.txt
```

### Model Parameters 
```python
vocab_size = 40000
embedding_dim = 16
max_length = 250
trunc_type = 'post'
oov_tok = '<OOV>'
padding_type = 'post'
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(x_train)
word_index = tokenizer.word_index

```

You can find the full code at my repositories.
## Authors

- [@ShivamJhunjhunwala](https://github.com/ShivamJhunjhunwala)


## 🚀 About Me
I'm a [@ShivamJhunjhunwala](https://github.com/ShivamJhunjhunwala)

I am coder, learner, developer, tester and many more. 
I am a keen learner trying to learn and implement new things everyday. To see more of my work please visit my other repositories.
## 🛠 Skills
Languages
* Javascript
* HTML
* CSS 
* Java 
* Python 
* SQL
Machine Learning and Data Science
* Data Science 
* Machine Learning 
* Deep Neural Network
* Natural language Processing
* Image Processing

## Other Common Github Profile Sections
👩‍💻 I'm currently working on Supervised Machine Learning Problem

🧠 I'm currently learning Data Science using Python

👯‍♀️ I'm looking to collaborate on Supervised Machine Learning Problem using python and sql

💬 Ask me about SQl, Python, Animes, F1, Cricket

📫 Reach me at shivamjhunjhunwala42@gmail.com


## 🔗 Links

Visit my social profiles to get to know more about me

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/shivam-j-ba3b99b5/)

[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/ShivamJhunjhu42)


## Appendix

For any such help needed in SQL scripting reach out to me for code explanation and premade and custom SQL codes for development and production usage.


## Feedback

If you have any feedback, please reach out to me at shivamjhunjhunwala42@gmail.com


## Support

For support, email shivamjhunjhunwala42@gmail.com or join my telegram at [@ShivamJhunjhunwala](https://t.me/SOFTWARE_DEV_ENG_PYTHON).

