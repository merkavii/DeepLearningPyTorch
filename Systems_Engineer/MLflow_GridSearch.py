import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import itertools

# * ==============================================================================
# *                          تنظیمات اولیه MLflow
# * ==============================================================================
# ? تعیین نام آزمایش (Experiment) برای گروه‌بندی اجراهای GridSearch
mlflow.set_experiment("ticket_classifier_grid_search")

# ? تعریف ساختار مدل شبکه عصبی دسته‌بندی تیکت
class TicketClassifier(nn.Module):
    def __init__(self, vocab_size=3000, num_classes=3):
        super().__init__()
        self.classifier = nn.Linear(vocab_size, num_classes) 

    def forward(self, x):
        return self.classifier(x)

# * ==============================================================================
# *                 تعریف فضای جستجوی GridSearch (Grid Space)
# * ==============================================================================
# ? مقادیری که می‌خواهیم GridSearch روی آن‌ها تست انجام دهد
param_grid = {
    'learning_rate': [0.01, 0.001],
    'batch_size': [16, 32],
    'epochs': [5]
}

# ? ساخت تمام ترکیب‌های ممکن از هایپرپارامترها (Cartesian Product)
# ! مثال: (0.01, 16, 5), (0.01, 32, 5), (0.001, 16, 5), ...
keys, values = zip(*param_grid.items())
grid_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

vocab_size = 3000
num_classes = 3

print(f"Total GridSearch combinations: {len(grid_combinations)}")

# * ==============================================================================
# *                 حلقه اصلی GridSearch همراه با MLflow
# * ==============================================================================
for i, params in enumerate(grid_combinations):
    
    # ? استخراج مقادیر فعلی ترکیب GridSearch
    lr = params['learning_rate']
    batch_size = params['batch_size']
    epochs = params['epochs']
    
    run_name = f"grid_run_{i+1}_lr_{lr}_bs_{batch_size}"
    
    # * هر ترکیب GridSearch تحت یک Run مستقل در MLflow ذخیره می‌شود
    with mlflow.start_run(run_name=run_name):
        
        # * ----------------------------------------------------------------------
        # * ۱. ثبت هایپرپارامترهای این ترکیب در MLflow
        # * ----------------------------------------------------------------------
        mlflow.log_params(params)
        mlflow.log_param("vocab_size", vocab_size)

        # ? ساخت مدل، تابع زیان (Loss) و Optimizer جدید برای این اجرا
        model = TicketClassifier(vocab_size=vocab_size, num_classes=num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # * ----------------------------------------------------------------------
        # * ۲. حلقه آموزش (Training Loop)
        # * ----------------------------------------------------------------------
        for epoch in range(epochs):
            # // inputs = torch.randn(batch_size, vocab_size) <-- دیتای فرضی بردارهای متنی
            inputs = torch.randn(batch_size, vocab_size)
            targets = torch.randint(0, num_classes, (batch_size,))

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # ? ثبت مقدار Loss در هر دوره برای رسم نمودار مقایسه‌ای در MLflow
            mlflow.log_metric("train_loss", loss.item(), step=epoch)

        # * ----------------------------------------------------------------------
        # * ۳. ثبت متریف نهایی و ذخیره مدل
        # * ----------------------------------------------------------------------
        # ? ثبت Loss نهایی این ترکیب جهت سورت کردن اجراها در UI
        mlflow.log_metric("final_loss", loss.item())
        
        # ! ذخیره فایل مدل آموزش‌دیده مربوط به این ترکیب خاص
        input_example = torch.randn(1, 3000)
        mlflow.pytorch.log_model(model, 
                                name='model',
                                input_example=input_example,
                                serialization_format='pickle')
        
        print(f"Run {i+1}/{len(grid_combinations)} completed (LR={lr}, BS={batch_size}) - Final Loss: {loss.item():.4f}")

# todo: پس از اتمام، دستور mlflow ui را بزن تا تمام اجراها را کنار هم مقایسه کنی
print("\nGridSearch completed successfully! All runs are logged in MLflow.")