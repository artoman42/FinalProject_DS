# FinalProject_DS
Final project of EPAM Data Science Program

## EDA part 
###
- Research is made only on 80% of training data, because i thought, that test data has no labels, but then i remake it on full training data - Research_2 (They are mostly equals)

### Conclusions from EDA

- Training data consists of 40000 examples
- Target variable is equally distributed(positive - 50% vs negative 50%)
- Also, there is no omissions in training data.
- I thought about feature like CAPS words, i thought that, they can meaningful impact, like very emotional opinion, and can contribute to model. But after analysis, it gave results, that first of all, more positive reviews contains CAPS words, and after deeper dive in data, i see, that some abbreviation, that doesn't contain meaningful sentimental opinion. And that some CAPS words, can have 2 or more meanings, like real and sarcastic. So i make a decision, that in this task better to just lower case, and it gave meaningful results.

### Preprocessing description

- <b>Removing duplicates</b> (for training only): Eliminates duplicate entries from the training data to prevent biasing the model towards specific instances and ensure a more balanced representation of sentiments.

- <b>Removing stopword</b>s: Removes common words (such as "the", "is", "and") that do not carry significant meaning and could potentially add noise to the analysis.

- <b>Making lower casing</b>: Converts all text to lowercase to ensure consistency in word representations and reduce the vocabulary size by collapsing words with different cases into the same token.

- <b>Removing punctuation</b>: Eliminates punctuation marks from the text, as they typically do not contribute to sentiment analysis and can interfere with tokenization and subsequent analysis.

- <b>Removing URLs</b>: Strips out URLs, which often contain irrelevant information and could mislead the sentiment analysis model if included.

- <b>Removing HTML tags</b>: Filters out HTML tags that may be present in text data, especially when dealing with scraped or web-based content, to ensure that only the textual content is considered.

- <b>Removing emojis and emoticons</b>: Eliminates emojis (graphical symbols representing emotions or concepts) and emoticons (textual representations of facial expressions) as they may not carry consistent sentiment across different contexts and could introduce noise to the analysis.

- <b>Word tokenization</b>: Splits the text into individual words or tokens, which are the basic units of analysis for natural language processing tasks.

- <b>Lemmatizing words</b>: Reduces inflected words to their base or dictionary form (lemma) to normalize variations of the same word and improve the model's ability to recognize semantic similarities.

- <b>TF-IDF vectorizing</b>: Converts the preprocessed text into numerical representations using the TF-IDF (Term Frequency-Inverse Document Frequency) technique, which assigns weights to words based on their frequency in the document and inverse frequency across the entire corpus. This step transforms the text data into a format suitable for machine learning algorithms, where each document is represented as a vector in a high-dimensional space.

`P.S.` Analysis showed, that training data doesn't contain html tags, urls or emoticons. But i think, it is good practice to do it in preprocessing pipeline. So we can be sure in training, and especially in inference data.

### Reasonings on model selection
- Firstly i chose lemmatization over stemming in text preprocessing, because it gave better results, and took lower time.
- I chose Tfidfvectorizer, because it gave better results, and models train faster with him much faster, than with CountVectorizer
- For final model, i chose LogisticRegression, because it achieved the best accuracy result, and had one of the lowest training time.
### Overall perfomance evaluation
- Training preprocessing take in average 250s, Test - 60s
- Model training tooks 3-4s, and give ~90%(89.54) accuracy on test data
- Testing also takes just a few seconds
### Potential business applications and value for business
1. **Movie Studios and Production Companies**:
   - *Audience Feedback Analysis*: Gain insights into audience sentiments towards specific movies, allowing studios to understand what aspects of a film resonate positively or negatively with viewers.
   - *Market Research*: Use sentiment analysis to assess the potential success of upcoming movie releases, guiding decisions on marketing strategies and budget allocations.
   - *Script and Plot Optimization*: Analyze sentiment trends to identify patterns in successful movie plots and character development, helping studios refine scripts and enhance audience engagement.

2. **Streaming Platforms**:
   - *Content Curation and Recommendation*: Incorporate sentiment analysis into recommendation algorithms to suggest movies that align with users' preferences and mood, improving user satisfaction and retention.
   - *Content Quality Assessment*: Evaluate the overall sentiment of user reviews to gauge the quality and popularity of movies available on the platform, informing decisions on content acquisition and licensing.

3. **Movie Theaters and Exhibition Chains**:
   - *Audience Experience Enhancement*: Monitor sentiment trends to understand patrons' experiences in theaters, identifying areas for improvement such as sound quality, seating comfort, or concession offerings.
   - *Programming Decision Support*: Use sentiment analysis to predict audience turnout and reception for different movies, optimizing scheduling and programming decisions to maximize ticket sales.

4. **Market Research Firms**:
   - *Consumer Insights*: Provide clients in the entertainment industry with valuable insights into consumer preferences, attitudes, and behavior regarding popular movies, enabling targeted marketing campaigns and product development strategies.
   - *Competitive Analysis*: Analyze sentiment data to benchmark the performance of movies against competitors, identifying strengths and weaknesses to inform competitive positioning and marketing strategies.

5. **Advertising and Marketing Agencies**:
   - *Campaign Performance Evaluation*: Assess the effectiveness of movie advertising campaigns by tracking sentiment changes before, during, and after promotional activities, enabling real-time adjustments and optimization.
   - *Influencer Marketing*: Identify influencers and opinion leaders within the movie-going community by analyzing sentiment patterns in user-generated content, facilitating targeted partnerships and endorsement strategies.

6. **Retail and Merchandising**:
   - *Licensed Merchandise Sales*: Analyze sentiment data to identify popular movie franchises and characters, informing decisions on product licensing and merchandising opportunities.
   - *Promotional Tie-Ins*: Leverage sentiment insights to tailor promotional campaigns and product offerings that resonate with fans of popular movies, driving engagement and sales in retail outlets.

## MLE part

### Forking and Cloning from GitHub
To start using this project, you first need to create a copy on your own GitHub account by 'forking' it. On the main page of the `FinalProject_DS` project, click on the 'Fork' button at the top right corner. This will create a copy of the project under your own account. You can then 'clone' it to your local machine for personal use. To do this, click the 'Code' button on your forked repository, copy the provided link, and use the `git clone` command in your terminal followed by the copied link. This will create a local copy of the repository on your machine, and you're ready to start!

### Setting Up Development Environment

To set up your development environment, follow these steps:

1. **Install Python 3.9**: If you haven't already, download and install Python 3.9 from the [official Python website](https://www.python.org/downloads/release/python-3916/). Ensure that you select the appropriate installer for your operating system.

2. **Download Visual Studio Code (VSCode)**: VSCode is an excellent IDE for Python development. You can download it from the [official website](https://code.visualstudio.com/Download).

3. **Install VSCode**: After downloading the installer, follow the installation instructions provided for your operating system. Once installed, open VSCode.

4. **Open Workspace**: Navigate to the directory where you have cloned the forked repository for your project. In VSCode, go to the `File` menu and click `Add Folder to Workspace`. Select the directory where your project resides and add it to the workspace.

5. **Configure Python Interpreter**: In VSCode, open the command palette by pressing `Ctrl + Shift + P` (or `Cmd + Shift + P` on macOS) and type `Python: Select Interpreter`. Choose Python 3.9.x from the list of available interpreters.

6. **Start Coding**: VSCode provides various features such as syntax highlighting, code completion, and debugging configurations for Python development. You can now edit files, navigate through your project, and start contributing to your project.

7. **Running Scripts**: To execute Python scripts, open a new terminal in VSCode by selecting `Terminal -> New Terminal` from the menu. Navigate to the directory where your Python script is located and run it using the `python` command followed by the script filename.

With Python 3.9.x installed and your development environment configured in VSCode, you're all set to begin coding and contributing to your project. Happy coding!

### Installing Docker Desktop

Installing Docker Desktop is a straightforward process. Head over to the Docker official website's download page ([Docker Download Page](https://www.docker.com/products/docker-desktop)), and select the version for your operating system - Docker Desktop is available for both Windows and Mac. After downloading the installer, run it, and follow the on-screen instructions. 

Once the installation is completed, you can open Docker Desktop to confirm it's running correctly. It will typically show up in your applications or programs list. After launching, Docker Desktop will be idle until you run Docker commands. This application effectively wraps the Docker command line and simplifies many operations for you, making it easier to manage containers, images, and networks directly from your desktop. 

Keep in mind that Docker requires you to have virtualization enabled in your system's BIOS settings. If you encounter issues, please verify your virtualization settings, or refer to Docker's installation troubleshooting guide. Now you're prepared to work with Dockerized applications!

## Data:
Data is the cornerstone of any Machine Learning project. For uploading the data, use the script located at `src/data_process/data_generation.py`. It has also 3 flags `--mode training` for only training data download, `--mode inference` for only testing data download. In same way works `src/data_process/data_processor.py` it uses downloaded data, process it and store it to `data/processed/`.
## Training:
Here training pipeline is done in docker, so it is easier to train with it. If you wanna run it locally, you need to run a few scripts sequentlly(detailed about it after docker)

1. To train the model using Docker: 

- Build the training Docker image. If the built is successfully done, it will automatically train the model:
```bash
docker build -f ./src/train/Dockerfile --build-arg settings_name=settings.json -t training_image .
```
- Then you should run the container with the following parameters to ensure that the trained model is here:
```bash
docker run -dit training_image
```
For moving model, vectorizer, and test_metrics , you also need to create folder `outputs/models`, `outputs/vectorizers`  on your machine manually.
Then, move the trained model, vectorizer from the directories inside the Docker container to the local machine using:
```bash
docker cp <container_id>:/app/outputs/models/<model_name>.pkl ./outputs/models
```
```bash
docker cp <container_id>:/app/outputs/vectorizers/<vectorizer_name>.pkl ./outputs/vectorizers
```

Replace `<container_id>` with your running Docker container ID and `<model_name>.pkl` with your model's name and `<vectorizer_name>.pkl` with your vectorizer`s.
2. Alternatively you can run all locally with
```bash
python ./src/data_process/data_loader.py --mode training ; python ./src/data_process/data_processor.py --mode training ; python ./src/train/train.py

```
## Inference 
Once a model and vectorizer has been trained, it can be used to make predictions on new data in the inference stage. The inference stage is implemented in `inference/run.py`.
1. To run the inference using Docker, use the following commands:

- Build the inference Docker image:
```bash
docker build -f ./src/inference/Dockerfile --build-arg model_name=<model_name>.pkl --build-arg vectorizer_name=<vectorizer_name>.pkl --build-arg settings_name=settings.json -t inference_image .
```
- Then you should run the container with this command:
```bash
docker run -dit inference_image  
```
- Then for copy results you need to use this
```bash
docker cp <container_id>:/app/outputs/predictions/ ./outputs/
```
2. Alternatively you can run all locally with
```bash
python ./src/data_process/data_loader.py --mode inference ; python ./src/data_process/data_processor.py --mode inference ; python ./src/inference/inference.py
```
