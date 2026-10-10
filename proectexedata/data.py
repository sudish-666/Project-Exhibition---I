import kagglehub
import shutil
import os

path = kagglehub.dataset_download("rohanrao/air-quality-data-in-india")
print("Dataset downloaded to:", path)

src = os.path.join(path, "city_day.csv")
dst = "city_day.csv"
shutil.copy(src, dst)
print("Copied city_day.csv into your project folder.")