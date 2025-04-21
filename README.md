## 1. Create a virual environment :
```{bash}
conda create -n env_name
```

## 2. Activate the virtual environment :
```{bash}
conda activate env_name
```

## 3. Install the dependencies :
Run the following code in the root directory of the project :
```{bash}
pip install -r requirements.txt
```

## 4. Run the following code to start up the application :
```{bash}
streamlit run --server.fileWatcherType none app.py
```

Here, I had to turn off the filewatcher to fix a build issue due to incompatibility with either of streamlit or pytorch versions.

I have uncommented the following in app.py for now, to disable the filewatcher :

```{python}
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
```

So, we can directly run using :
```{bash}
streamlit run app.py
```

However, if it is commented, we will have to use the 1st command with fileWatcherType none.