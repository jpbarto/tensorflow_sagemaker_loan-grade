import os
import json
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.estimator.export.export import build_raw_serving_input_receiver_fn
from tensorflow.python.estimator.export.export_output import PredictOutput

logger = tf.logging

train_data = None # dict containing 'features' and 'labels' which store ndarrays
eval_data = None

# columns to extract from the CSV
APPLICANT_NUMERIC = ['annual_inc', 'dti', 'age_earliest_cr', 'loan_amnt', 'installment']
APPLICANT_CATEGORICAL = ['application_type', 'home_ownership', 'addr_state', 'term']
CREDIT_NUMERIC = ['acc_now_delinq', 'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy',
                  'bc_util', 'delinq_2yrs', 'delinq_amnt', 'fico_range_high', 'fico_range_low',
                  'last_fico_range_high', 'last_fico_range_low', 'open_acc', 'pub_rec', 'revol_util',
                  'revol_bal', 'tot_coll_amt', 'tot_cur_bal', 'total_acc', 'total_rev_hi_lim',
                  'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats',
                  'num_bc_tl', 'num_il_tl', 'num_rev_tl_bal_gt_0', 'pct_tl_nvr_dlq',
                  'percent_bc_gt_75', 'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit',
                  'total_il_high_credit_limit', 'all_util', 'loan_to_income',
                  'installment_pct_inc', 'il_util', 'il_util_ex_mort', 'total_bal_il', 'total_cu_tl']

FEATURES = APPLICANT_NUMERIC + APPLICANT_CATEGORICAL + CREDIT_NUMERIC
LABEL = 'grade'
COLUMNS = FEATURES + [LABEL]

INPUT_TENSOR_NAME = "inputs"

# Model definition for Tensorflow Estimator.  Creates a model, defines a loss function, and produces an EstimatorSpec
# for training by TensorFlow
def model_fn(features, labels, mode, hyperparameters = {}):
    """
    Implement code to do the following:
    1. Configure the model with TensorFlow operations
    2. Define the loss function for training/evaluation
    3. Define the training operation/optimizer
    4. Generate predictions
    5. Return predictions/loss/train_op/eval_metric_ops in EstimatorSpec object 

    For more information on how to create a model_fn, see 
    https://www.tensorflow.org/extend/estimators#constructing_the_model_fn.
    
    Args:
        features:  A dict containing the features passed to the model with train_input_fn in 
           training mode, with eval_input_fn in evaluation mode, and with serving_input_fn 
           in predict mode.
        labels:    Tensor containing the labels passed to the model with train_input_fn in 
           training mode and eval_input_fn in evaluation mode. It is empty for 
           predict mode.
        mode:     One of the following tf.estimator.ModeKeys string values indicating the context 
           in which the model_fn was invoked: 
           - TRAIN: The model_fn was invoked in training mode. 
           - EVAL: The model_fn was invoked in evaluation mode. 
           - PREDICT: The model_fn was invoked in predict mode:

        hyperparameters: The hyperparameters passed to your Amazon SageMaker TrainingJob that 
           runs your TensorFlow training script. You can use this to pass hyperparameters 
           to your training script. 
                          
    Returns: An EstimatorSpec, which contains evaluation and loss function. 
    """
    params = hyperparameters
    
    logger.info ("Creating EstimatorSpec...")
    
    learning_rate = 0.001
    if 'learning_rate' in params:
        learning_rate = params['learning_rate']
        
    logger.info ('Generating layers with input {}...'.format (features[INPUT_TENSOR_NAME]))
    logger.info ('Using a learning rate of {}'.format (learning_rate))
    
    with tf.name_scope ('input_layer'):
        layer1 = tf.layers.dense(features[INPUT_TENSOR_NAME], 100, activation=tf.nn.relu, kernel_constraint = tf.keras.constraints.max_norm (3), name = 'loan_features')
        layer2 = tf.layers.dropout (layer1, rate = 0.2, name = 'loan_features_drop')
    with tf.name_scope ('hidden_layer'):
        layer3 = tf.layers.dense (layer2, 60, activation = tf.nn.relu, kernel_constraint = tf.keras.constraints.max_norm (3), name = 'middle_layer')
        layer4 = tf.layers.dropout (layer3, rate = 0.2, name = 'middle_layer_drop')
    with tf.name_scope ('output_layer'):
        logits = tf.layers.dense (inputs = layer4, units = 7, name = 'logits')
    
    logger.debug ("Output layer: {}".format (logits))
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input = logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    
    logger.debug ("Labels data type: {}".format (labels))
    logger.debug ("Predictions: {}".format (predictions))
    
    logger.info ('Test for predict mode...')
    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
        logger.info ('Returning prediction EstimatorSpec')
        logger.info ("Prediction classes are {}".format (predictions['classes']))
        logger.info ("Prediction probs are {}".format (predictions['probabilities']))
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions = predictions,
            export_outputs={"grade_prediction": PredictOutput(predictions)})
    
    # 2. Define the loss function for training/evaluation using Tensorflow.
    m_labels = labels
    m_predictions = predictions
    
    logger.info ('Generate loss function...')
    with tf.name_scope ('cross_entropy_loss'):
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels = tf.cast (labels, tf.int32), 
            logits=logits
        )
        tf.summary.scalar ('x_ent_loss', loss)
    
    logger.info ('Test for train mode...')
    if mode == tf.estimator.ModeKeys.TRAIN:
        with tf.name_scope ('trainer'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
            train_op = optimizer.minimize(
                loss = loss,
                global_step = tf.train.get_global_step()
            )
        
        logger.info ("Returning EstimatorSpec for TRAIN mode")
        return tf.estimator.EstimatorSpec(
            mode=mode, 
            loss=loss, 
            train_op=train_op
        )
    
    logger.info ('Generate evaluation metrics...')
    # Add evaluation metrics (for EVAL mode)
    with tf.name_scope ('accuracy'):
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels = tf.argmax (input = labels, axis = 1), 
                predictions=predictions["classes"]
            )
        }
        tf.summary.scalar ('accuracy', eval_metric_ops['accuracy'])
    
    logger.info ("Returning EstimatorSpec")
    return tf.estimator.EstimatorSpec(
        mode=mode, 
        loss=loss, 
        eval_metric_ops=eval_metric_ops
    )
    
 
def train_input_fn (training_dir, hyperparameters):
    """
    Implement code to do the following:
    1. Read the **training** dataset files located in training_dir
    2. Preprocess the dataset
    3. Return 1) a mapping of feature columns to Tensors with
    the corresponding feature data, and 2) a Tensor containing labels
 
    For more information on how to create a input_fn, see https://www.tensorflow.org/get_started/input_fn.

    Args:
        training_dir:    Directory where the dataset is located inside the container.
        hyperparameters: The hyperparameters passed to your Amazon SageMaker TrainingJob that 
           runs your TensorFlow training script. You can use this to pass hyperparameters 
           to your training script.
 
    Returns: (data, labels) tuple
    """
    logger.info ("Generating training input function...")
    
    if train_data is None:
        read_csv_data (training_dir)
        
    logger.info ("Returning input function")
    return gen_input_fn (train_data, epochs = None, shuffle_flag = True)
        
def eval_input_fn(training_dir, hyperparameters):
    """
   Implement code to do the following:
    1. Read the **evaluation** dataset files located in training_dir
    2. Preprocess the dataset
    3. Return 1) a mapping of feature columns to Tensors with
    the corresponding feature data, and 2) a Tensor containing labels
 
    For more information on how to create a input_fn, see https://www.tensorflow.org/get_started/input_fn.

    Args:
     training_dir: The directory where the dataset is located inside the container.
     hyperparameters: The hyperparameters passed to your Amazon SageMaker TrainingJob that 
           runs your TensorFlow training script. You can use this to pass hyperparameters 
           to your training script.
 
    Returns: (data, labels) tuple
    """
    logger.info ("Generating evaluation input function...")
    
    if eval_data is None:
        read_csv_data (training_dir)
        
    logger.info ("Returning input function")
    return gen_input_fn (eval_data, epochs = 1, shuffle_flag = False)

 
def serving_input_fn(hyperparameters):
    """
    During training, a train_input_fn() ingests data and prepares it for use by the model. 
    At the end of training, similarly, a serving_input_fn() is called to create the model that 
    will be exported for Tensorflow Serving.

	Use this function to do the following:

		- Add placeholders to the graph that the serving system will feed with inference requests.
		- Add any additional operations needed to convert data from the input format into the 
         feature Tensors expected by the model.

	The function returns a tf.estimator.export.ServingInputReceiver object, which packages the placeholders
      and the resulting feature Tensors together.

	Typically, inference requests arrive in the form of serialized tf.Examples, so the 
      serving_input_receiver_fn() creates a single string placeholder to receive them. The serving_input_receiver_fn() 
      is then also responsible for parsing the tf.Examples by adding a tf.parse_example operation to the graph.

	For more information on how to create a serving_input_fn, see 
      https://github.com/tensorflow/tensorflow/blob/18003982ff9c809ab8e9b76dd4c9b9ebc795f4b8/tensorflow/docs_src/programmers_guide/saved_model.md#preparing-serving-inputs.
	
    Args:	
     hyperparameters: The hyperparameters passed to your TensorFlow Amazon SageMaker estimator that 
           deployed your TensorFlow inference script. You can use this to pass hyperparameters 
           to your inference script.

	"""
    logger.info ("Generating input function for serving...")
    
    feature_spec = {INPUT_TENSOR_NAME: tf.placeholder (dtype=tf.float32, shape=[1, 101])}
    
    logger.info ("Returning input function")
    # return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)()
    return tf.estimator.export.build_raw_serving_input_receiver_fn (feature_spec) ()

# for training data epochs should be None and shuffle_flag set to True
# for evaluation or prediction data epochs should be 1 and shuffle_flag set to False
def gen_input_fn(data, epochs = 1, shuffle_flag = False):   
    return tf.estimator.inputs.numpy_input_fn(
            x = {INPUT_TENSOR_NAME: data['features']},
            y = data['labels'],
            num_epochs = epochs,
            shuffle = shuffle_flag) ()

# Parse the data CSV and return Pandas DataFrames for training / evaluation / and prediction in an 60 / 20 / 20 split
def read_csv_data (training_dir): 
    global train_data, eval_data
    
    grade_categories = [g for g in "ABCDEFG"]
    
    logger.info ("Reading training data from {}".format (os.path.join (training_dir, 'train_data.csv')))
    
    lg_data = pd.read_csv (os.path.join (training_dir, 'train_data.csv'), usecols = COLUMNS)
    lg_data['grade'] = lg_data['grade'].astype ('category', categories = grade_categories, ordered = True)
    # shuffle the data set
    lg_data = lg_data.sample (frac = 1, random_state = 2501)
    
    bad_rows = lg_data.isnull ().T.any ().T.sum ()
    if bad_rows > 0:
        logger.debug("Rows with null/NaN values: {}".format(bad_rows))
        logger.debug("Columns with null/NaN values:")
        logger.debug(pd.isnull(lg_data).sum() > 0)
        logger.debug("Dropping bad rows...")
        lg_data.dropna(axis=0, how='any', inplace=True)
        logger.debug("Rows with null/NaN values: {}".format(lg_data.isnull().T.any().T.sum()))
        
    # Subset to get feature data
    x_df = lg_data.loc[:, APPLICANT_NUMERIC + CREDIT_NUMERIC + APPLICANT_CATEGORICAL]

    # Update our X dataframe with categorical values replaced by one-hot encoded values
    for col in APPLICANT_CATEGORICAL:
        # use get_dummies() to do one hot encoding of categorical column
        x_df = x_df.merge(pd.get_dummies(x_df[col]), left_index=True, right_index=True)
        
        # drop the original categorical column
        x_df.drop(col, axis=1, inplace=True)
    
    # Ensure all numeric features are on the same scale
    for col in APPLICANT_NUMERIC + CREDIT_NUMERIC:
        x_df[col] = (x_df[col] - x_df[col].mean()) / x_df[col].std()
    x_df = x_df.astype (np.float32)

    # Specify the target labels and flatten the array
    y = pd.get_dummies(lg_data[LABEL]).astype (np.float32)

    # Create train, eval, and test sets
    eval_start = int(x_df.shape[0]*.8)
    
    train_data = {
        'features': x_df.iloc[:eval_start, :].as_matrix (),
        'labels': y.iloc[:eval_start, :].as_matrix ()
    }
    eval_data = {
        'features': x_df.iloc[eval_start:, :].as_matrix (),
        'labels': y.iloc[eval_start:, :].as_matrix ()
    }
    logger.info ("Training data set feature shape {}, label shape {}, read complete".format (train_data['features'].shape, train_data['labels'].shape))
    
def no_input_fn (data, content_type):
    logger.info ("Processing content type {}, data {}".format (content_type, data))
    return tf.constant (json.loads (data))