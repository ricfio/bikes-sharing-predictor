from dataset import Dataset
from models.mplregressor import Model
#from models.dnnregressor import Model
import seaborn as sns
import matplotlib.pyplot as plt


def model_train(model, dataset):
    X_train, y_train = dataset.get_train()
    model.train(X_train, y_train)


def model_check(model, dataset):
    X_test, y_test = dataset.get_test()
    p_test = model.predict(X_test)
    model.print_error(y_test, p_test)

    residuals = p_test - y_test

    # sns.scatterplot(x=y_test, y=residuals)
    # plt.xlabel('bikes')
    # plt.ylabel('error')

    sns.scatterplot(x=X_test.index, y=y_test)
    sns.scatterplot(x=X_test.index, y=p_test, hue=residuals)
    plt.xlabel('index')
    plt.ylabel('prediction vs reality')

    # sns.scatterplot(x=y_test, y=residuals)

    # sns.scatterplot(x=df['season'], y=residuals)
    # sns.scatterplot(x=df['yr'], y=residuals)
    # sns.scatterplot(x=df['holiday'], y=residuals)
    # sns.scatterplot(x=df['weekday'], y=residuals)
    # sns.scatterplot(x=df['workingday'], y=residuals)
    # sns.scatterplot(x=df['weathersit'], y=residuals)
    # sns.scatterplot(x=df['temp'], y=residuals)
    # sns.scatterplot(x=df['atemp'], y=residuals)
    # sns.scatterplot(x=df['hum'], y=residuals)
    # sns.scatterplot(x=df['windspeed'], y=residuals)
    # sns.scatterplot(x=df['hr_sin'], y=residuals)
    # sns.scatterplot(x=df['hr_cos'], y=residuals)
    # sns.scatterplot(x=df['mm_sin'], y=residuals)
    # sns.scatterplot(x=df['mm_cos'], y=residuals)

    gcf = plt.gcf()
    gcf.canvas.set_window_title('Bikes regression (neural network)')

    plt.show()


def main():
    # Get dataset
    dataset = Dataset()
    # Get the model
    model = Model('mlpregressor-20201011')
    # model = Model('dnnpregressor-20201011')
    # Train
    # model_train(model, dataset)
    # Test
    model_check(model, dataset)


if __name__ == '__main__':
    main()
