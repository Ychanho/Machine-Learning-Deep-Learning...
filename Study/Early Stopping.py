training_epochs = 100
batch_size = 100

timestamp = str(int(time, time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir,"runs",timestamp))
checkpoint_dir = os.path.abspath(os.path.join(out_dir,"checkpotints"))
checkpoint_prefix = os.path.join(checkpoint_dir,"model")
if not os.path.exists(checkpoint_dir)
    os.makedirs(checkpoint_dir)
saver = tf.train.Saver(tf.global_variables(),max_to_keep=3)