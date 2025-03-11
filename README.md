# 11868_final_project
Repository for 11868 Final Project.

## Setup

1. **Create a Conda Environment**
   Use the following command to create and activate a new environment for the SFT training:
   
   ```bash
   conda create -n finetune python=3.10
   conda activate finetune
   ```
2. **Install Dependencies**
   After activating the environment, install all required dependencies by running:
   
   ```bash
   pip install -r requirements.txt
   ```

## Testing

To run all tests, do

```
pytest
```

To run tests in a specific file, do either

```
python3 <path/to/filename>
```
For usual Python debugging or 

```
pytest <path/tol/filename>
```
To use the pytest suite.