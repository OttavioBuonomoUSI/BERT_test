import pandas as pd
from tqdm import tqdm

from data_preprocessing.DataManage import DataManage
from models.BertCustomizer import BertCustomizer
import matplotlib.pyplot as plt
from train import get_df
import os
import sys

if __name__ == '__main__':
    os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "UNCOMPRESSED"

    df_complete = get_df()
    # get all issues with refIssue not equal to null
    # df = df_complete[df_complete['RefIssue'].notnull()]

    # RefIssues
    df_test = df_complete[df_complete["RefIssue"].notnull()]["RefIssue"].rename("IssueId").to_frame()
    df_test = df_test.reset_index()
    df_test = df_test.drop("index", axis=1)
    df_test_2 = df_complete[df_complete["IssueId"].astype(str).isin(df_test["IssueId"].values)][["IssueId", "AssigneeLogin"]]
    df_test_2 = df_test_2.rename(columns={"AssigneeLogin": "RefAssignee"})
    df_final = df_complete.merge(df_test_2, on="IssueId", how="left")

    # Workload
    if not os.path.exists("./db/workload.csv"):
        print("WORKLOAD CALCULATOR")
        for index in tqdm(range(df_complete.shape[0])):
            if index == 0:
                df_workload_complete = []
            else:
                actual_date = df_complete.iloc[index]["CreatedAt"]
                df_selected_issues = df_complete.iloc[:index]
                df_workload = {}
                for assignee in list(set(df_complete["AssigneeLogin"])):
                    condition1 = df_selected_issues["ClosedAt"] >= actual_date
                    condition2 = df_selected_issues["AssigneeLogin"] == assignee
                    a = str(assignee) + "_Wl"
                    df_workload[a] = df_selected_issues[condition1 & condition2].shape[0]
                df_workload_complete.append(df_workload)
        workload = pd.DataFrame.from_dict(df_workload_complete)
        workload.to_csv("./db/workload.csv")
    else:
        workload = pd.read_csv("./db/workload.csv")

    workload_cols_name = list(workload.columns)
    df_final2 = pd.concat([df_final, workload], axis=1)

    selected_variables = ['CleanText', 'AssigneeLogin', 'RefAssignee', 'CreatedAt'] + workload_cols_name
    df_final2["CreatedAt"] = pd.to_numeric(pd.to_datetime(df_final2["CreatedAt"]))
    df = df_final2[selected_variables]

    # Data Ingestion
    data_manage = DataManage(
        data_frame=df,
        text_col_name="CleanText",
        label_col_name="AssigneeLogin",
        random_split_method=False,
        test_percentage=20,
        val_percentage=20
    )

    # On Entire DataFrame
    data_manage.split(type_check=False, show_info=True)
    data_manage.transform()

    data_manage.remove_html_tags()
    print("\n\n\nAFTER REMOVE HTML TAGS")
    print(data_manage.X_train_text)

    # On Textual Variable
    data_manage.remove_special_characters()
    print("\n\n\nAFTER SPECIAL")
    print(data_manage.X_train_text_pp)

    data_manage.remove_stopwords()
    print("\n\n\nAFTER STOPWORDS")
    print(data_manage.X_train_text_pp)

    data_manage.lemmatization()
    print("\n\n\nAFTER LEMMATIZATION")
    print(data_manage.X_train_text_pp)

    data_manage.to_lowercase()
    print("\n\n\nAFTER LOWERCASE")
    print(data_manage.X_train_text_pp)

    EPOCHS = 20
    BATCH_SIZE = 32
    steps_per_epoch = data_manage.X_train_text_pp.shape[0] // BATCH_SIZE
    num_train_steps = steps_per_epoch * EPOCHS
    num_warmup_steps = int(0.1 * num_train_steps)

    # Model
    bert_customizer = BertCustomizer(bert_model_name="bert_en_uncased_L-12_H-768_A-12",
                                     info=data_manage.info,
                                     show_info=True,
                                     batch_size=BATCH_SIZE)
    print("\n\n\ndata_manage.info")
    print(data_manage.info)
    print("\n\n\ndata_manage.info['additional_input_count']")
    print(data_manage.info["additional_input_count"])

    bert_customizer.build(bert_trainability=True)
    bert_customizer.compile(num_train_steps=num_train_steps, num_warmup_steps=num_warmup_steps)

    train_generator = bert_customizer.get_tf_dataset_from_generator(
        text=data_manage.X_train_text_pp.apply(lambda x: ' '.join(x)),
        X=data_manage.X_train_pp,
        y=data_manage.y_train_pp)

    val_generator = bert_customizer.get_tf_dataset_from_generator(
        text=data_manage.X_val_text_pp.apply(lambda x: ' '.join(x)),
        X=data_manage.X_val_pp,
        y=data_manage.y_val_pp)

    history = bert_customizer.model.fit(train_generator,
                                        validation_data=val_generator,
                                        epochs=EPOCHS)

    history_dict = history.history
    print(history_dict.keys())

    acc = history_dict['categorical_accuracy']
    val_acc = history_dict['val_categorical_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)
    fig = plt.figure(figsize=(10, 6))
    fig.tight_layout()

    plt.subplot(2, 1, 1)
    # r is for "solid red line"
    plt.plot(epochs, loss, 'r', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig("./models/bert_history.png")
