# big-data-reu
Repository for 2021 Team 1 project done during the Online Interdisciplinary Big Data Analytics in Science and Engineering REU (Research Experience for Undergraduates) Site program at UMBC. Additional information can be found at https://bigdatareu.umbc.edu/summer2021/

**Title:** Big Data and Machine Learning Techniques for Sea Ice Prediction

**Team Members:** Eliot Kim, Peter Kruse, Skylar Lama, Jamal Bourne Jr, Michael Hu

**Mentors:** Sahara Ali, Jianwu Wang

**Colaborator:** Yiyi Huang, Research Scientist at NASA Langley Research Center and Adjunct Research Assistant Professor at UMBC

**Abstract:** Important natural resources in the Arctic rely heavily on sea ice making it important to forecast Arctic sea ice changes. Arctic sea ice forecasting often involves two connected tasks: sea ice concentration on each location and overall sea ice extent. Instead of having two separate models for two forecasting tasks, in this paper, we study how to use multi-task learning techniques and leverage the connections between ice concentration and ice extent to improve accuracy for both prediction tasks. Because of the spatiotemporal nature of the data, we designed two novel multi-task learning models based on CNN and ConvLSTM, respectively. We developed a  custom loss function which applies teaches the models to ignore land pixels when making predictions. Our experiments show our models can have better accuracies than separate  models that predict sea ice extent and concentration separately, and our accuracies are better than or comparable with results in the state-of-the-art studies. 

### Navigating the Repository
The names of each directory in this repository correspond to their purpose. *Preprocessing* is where the train and test data sets for each model can be found as well as the code from which they were derived. *Modeling* contains all models of our CNN and ConvLSTMs as well as sample outputs from each model and a copy of the land mask that is used to filter out land values. *Analysis* contains all of the analysis materials used in this research including a vector auto regression model and a climatology analysis. Finally, *Evaluation* contains the data and code used to evaluate and post-process the results from each model. 
