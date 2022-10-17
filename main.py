from data_preprocessing.DataManage import DataManage
from models.BertCustomizer import BertCustomizer
import matplotlib.pyplot as plt
from train import get_df
import os
import sys

if __name__ == '__main__':

    df_complete = get_df()
    df = df_complete[['CleanText', 'AssigneeLogin']]
    
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

    # Model
    bert_customizer = BertCustomizer(bert_model_name="bert_en_uncased_L-12_H-768_A-12",
                                     info=data_manage.info,
                                     show_info=True)
    print("\n\n\ndata_manage.info")
    print(data_manage.info)
    print("\n\n\ndata_manage.info['additional_input_count']")
    print(data_manage.info["additional_input_count"])

    bert_customizer.build(bert_trainability=True)
    bert_customizer.compile()

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
                                        epochs=5)

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
