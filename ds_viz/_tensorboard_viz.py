'''
Still need to add capability for images ... Right now, this only works given labels.
'''
# import tensorflow as tf
# import numpy as np
# import os
# import glob
# from tensorflow.contrib.tensorboard.plugins import projector
# from sklearn.preprocessing import scale
#
#
# def tbviz(df, df_report=None, name='Time_Series', logdir='tblogs'):
#     '''
#     Visualize multidemensional data (up to about 50K rows, efficiently) on Tensorboard (PCA or TSNE).
#
#     Parameters
#     ----------
#     df: pd.DataFrame
#         This is the multi-dimensional data; *only* the multidimensional data.
#
#     df_report: pd.DataFrame
#         This is the dataframe of information that you'd like to investigate for each item in `df`. The index here must match the index for `df`.
#         An example could be a `df` of images, and `df_report` with columns like "image_name", "image_date", etc.
#
#     name: str
#         This is just the name of the project.
#
#     logdir: str
#         Name of log folder. Default is "tblogs".
#
#     Returns
#     -------
#     None. You just need to access Tensorboard using the command line.
#
#     '''
#
#     a = scale(df.values.astype(np.float))
#
#     dirs = [x[x.rfind('/')+1:] for x in glob.glob(os.getcwd() + '/*')]
#
#     if logdir not in dirs:
#         os.mkdir(logdir)
#
#     LOG_DIR = os.getcwd() + '/' + logdir
#     metadata_name = 'metadata.tsv'
#     data_name = name
#
#     embedding_var = tf.Variable(a, name=data_name)
#     summary_writer = tf.summary.FileWriter(LOG_DIR)
#     config = projector.ProjectorConfig()
#     embedding = config.embeddings.add()
#     embedding.tensor_name = embedding_var.name
#
#     # Specify where you find the metadata
#     embedding.metadata_path = metadata_name
#
#     # Say that you want to visualise the embeddings
#     projector.visualize_embeddings(summary_writer, config)
#
#     sess = tf.InteractiveSession()
#     sess.run(tf.global_variables_initializer())
#     saver = tf.train.Saver()
#     saver.save(sess, LOG_DIR + '/model.ckpt', 1)
#
#     if df_report is None:
#         df_report = df.iloc[:, :2]
#
#     with open(LOG_DIR + '/' + metadata_name, 'w') as f:
#         f.write('Index\t' + '\t'.join(df_report.columns) + '\n')
#         for index, data in zip(df_report.index, df_report.values):
#             f.write(str(index) + "\t" + '\t'.join([str(x) for x in data]) + "\n")
#
#     print("Make sure you've activated your Tensorflow virtual env with `conda activate YOUR_TSF_ENV`.")
#     print(f"Call `tensorboard --logdir={logdir}`, and then follow instructions to get to tensorboard.")