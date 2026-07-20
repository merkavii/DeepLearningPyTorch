# * ==============================================================================
# *                          MLflow : دستیار هوشمند MLOps
# * ==============================================================================
# ? چالش اصلی: هنگام آموزش مدل، هایپرپایپ‌لاین‌ها (Hyperparameters) مثل Learning Rate،
# ? تعداد لایه‌ها و Batch Size مدام تغییر می‌کنند.
# ! بدون MLflow: مجبوریم تمام تست‌ها را در دفترچه یا اکسل یادداشت کنیم تا کانفیگ برتر پیدا شود!

# * ------------------------------------------------------------------------------
# * کلیدی‌ترین قابلیت‌های MLflow
# * ------------------------------------------------------------------------------

# ? 1. MLflow Tracking:
# ?    ثبت اتوماتیک و دقیق تمام پارامترها، متریک‌های ارزیابی (Loss, Accuracy) و خروجی‌ها (Artifacts).

# ? 2. Model Registry:
# ?    ورژن‌بندی مدل‌های آموزش‌دیده (مثل v1, v2) برای شناسایی نسخه آماده تحویل به پروداکشن.

# ? 3. UI اختصاصی:
# ?    ارائه داشبورد وب حرفه‌ای جهت مقایسه بصری و نموداری اجراهای مختلف (Runs).

# todo: اتصال MLflow به کد پایتورچ و تست ثبت اجرا در داشبورد وب (http://127.0.0.1:5000)
# // mlflow.log_param("deprecated_param", True)  <-- این خط قدیمی است و نیازی به اجرا ندارد

import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch

# * ==============================================================================
# *                          تنظیمات اولیه MLflow
# * ==============================================================================
# ? تعیین نام آزمایش (Experiment) - گروه‌بندی اجراهای مرتبط با پردازش تیکت‌ها
mlflow.set_experiment("ticket_classification_system")

# ? تعریف شبکه عصبی برای دسته‌بندی متن تیکت‌ها (مثلاً: مالی، فنی، پیگیری سفارش)
class TicketClassifier(nn.Module):
    def __init__(self, vocab_size=3000, num_classes=3):
        super().__init__()
        # ? تبدیل بردار ویژگی‌های متن تیکت (۳۰۰۰ کلمه) به ۳ دسته پشتیبانی
        self.classifier = nn.Linear(vocab_size, num_classes) 

    def forward(self, x):
        return self.classifier(x)

# * ==============================================================================
# *                 شروع ثبت اجرا در MLflow (Run Session)
# * ==============================================================================
# ? ثبت یک اجرای جدید با نام مشخص برای تست نرخ یادگیری متفاوت
with mlflow.start_run(run_name="nlp_ticket_model_lr_0.005"):
    
    # * --------------------------------------------------------------------------
    # * ۱. تنظیم و ثبت هایپرپارامترها (Hyperparameters)
    # * --------------------------------------------------------------------------
    lr = 0.005
    batch_size = 64
    epochs = 12
    vocab_size = 3000
    
    # ? ثبت هایپرپارامترها در MLflow جهت مقایسه در اجراهای بعدی
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("vocab_size", vocab_size)

    # ? ساخت نمونه از مدل دسته‌بندی تیکت، تابع زیان و بهینه‌ساز
    model = TicketClassifier(vocab_size=vocab_size, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # * --------------------------------------------------------------------------
    # * ۲. حلقه آموزش (Training Loop) و ثبت متریك‌ها
    # * --------------------------------------------------------------------------
    for epoch in range(epochs):
        # // inputs = torch.randn(batch_size, vocab_size) <-- دیتای فرضی بردارهای متنی
        inputs = torch.randn(batch_size, vocab_size)
        targets = torch.randint(0, 3, (batch_size,))  # ۳ کلاس: ۰: مالی | ۱: فنی | ۲: پیگیری

        # ? صفر کردن گرادیان‌ها، پیش‌بینی دسته‌بندی و محاسبات پس‌انتشار
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # ? ثبت میزان لاس آموزش تیکت‌ها در هر دوره (Epoch) برای داشبورد MLflow
        mlflow.log_metric("train_loss", loss.item(), step=epoch)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

    # * --------------------------------------------------------------------------
    # * ۳. ذخیره‌سازی فایل مدل (Artifact Logging)
    # * --------------------------------------------------------------------------
    # ! مهم: مدل نهایی تشخیص تیکت در MLflow ثبت می‌شود تا سرویس پشتیبانی بتواند آن را فراخوانی کند
    input_example = torch.randn(1, 3000)
    mlflow.pytorch.log_model(model,
                             name="ticket_model",
                             input_example=input_example,
                             serialization_format='pickle')
    
    # todo: چک کردن مدل دسته‌بندی تیکت‌ها در مرورگر وب با دستور mlflow ui
    print("Ticket classification model trained and logged to MLflow successfully.")