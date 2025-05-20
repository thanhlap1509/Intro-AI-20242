# Intro-AI-20242
## Data preparation
Inside the config.py file, locate the dataset_used parameter and configure to the appropriate .csv file in [data_raw](https://github.com/thanhlap1509/Intro-AI-20242/tree/main/data_raw) folder.
```
dataset_used = "./data_raw/PRSA_Data_Aotizhongxin_20130301-20170228.csv"
```
then run the following command to process data:
```
python data_processor.py
```
## Training model
After setting the config, run:
```
python main.py
```
to train the model.

## Running demo
Run:
```
streamlit run UI_IntroAI.py
```
then enter your email to run the demo.
