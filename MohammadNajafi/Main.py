import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# -----------------------------
# ساخت دیتاست
# X = تعداد ساعت مطالعه
# Y = نمره امتحان
# -----------------------------

data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "score": [10, 12, 15, 18, 20, 23, 26, 30]
}

df = pd.DataFrame(data)

X = df[["study_hours"]]   # ورودی
Y = df["score"]           # خروجی

# -----------------------------
# ساخت و آموزش مدل رگرسیون خطی
# -----------------------------

model = LinearRegression()
model.fit(X, Y)

# -----------------------------
# تست مدل با مقدار جدید
# -----------------------------

new_hour = 9
predicted_score = model.predict([[new_hour]])

# -----------------------------
# نمایش خروجی
# -----------------------------

print("تعداد ساعت مطالعه:", new_hour)
print("نمره پیش‌بینی شده:", predicted_score[0])
