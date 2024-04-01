class MyLogReg():
    def __init__(self, **kwargs):
        kwargs.setdefault("num_folds", 5)
        kwargs.setdefault("max_iterations", 10) # trained through cv
        kwargs.setdefault("step_size", 0.0001) # trained through cv

        self.train_data = None
        self.train_labels = None

        self.coef_ = None
        self.intercept_ = None

        self.plotting_df = {}

        for key, value in kwargs.items():
            setattr(self, key, value)

    def fit( self, X, y ):
        self.train_data = X
        self.train_labels = y

        self.avg_loss = []

        # Create a dictionary to hold the results of the fitting
        results_dict = {}
        best_weights = {}

        # If input labels are 0/1 then make sure to convert labels to -1 and 1
        # for learning with the logistic loss.
        self.train_labels = np.where( self.train_labels == 1, 1, -1 )

        # Calculate folds
        fold_indicies = []

        self.plotting_dict = {
            "max_iterations": [],
            "avg_loss": []
        }

        # Pick random entries for validation/subtrain
        fold_vec = np.random.randint( low=0,
                                      high=self.num_folds,
                                      size=self.train_labels.size )

        # for each fold,
        for fold_number in range( self.num_folds ):
            subtrain_indicies = []
            validation_indicies = []
            # check if index goes into subtrain or validation list
            for index in range( len( self.train_data ) ):
                if fold_vec[index] == fold_number:
                    validation_indicies.append( index )
                else:
                    subtrain_indicies.append( index )

            fold_indicies.append( [ subtrain_indicies, validation_indicies ] )

        # Loop over the folds
        for foldnum, indicies in enumerate( fold_indicies ):
            # Get indicies of data chosen for this fold
            index_dict = dict( zip( [ "subtrain", "validation" ], indicies ) )
            set_data_dict = {}

            # Dictionary for test and train data
            for set_name, index_vec in index_dict.items():
                set_data_dict[ set_name ] = {
                    "X":self.train_data[ index_vec ],
                    "y":self.train_labels[ index_vec ]
                }

            # Define a variable called scaled_mat which has
            subtrain_data = set_data_dict[ "subtrain" ][ 'X' ]
            subtrain_labels = set_data_dict[ "subtrain" ][ 'y' ]

            scaled_mat = subtrain_data

            nrow, ncol = scaled_mat.shape

            learn_features = np.column_stack( [
                np.repeat( 1, nrow ),
                scaled_mat
            ] )

            weight_vec = np.zeros( ncol + 1 )

            subtrain_mean = subtrain_data.mean( axis = 0 )
            subtrain_sd = np.sqrt(subtrain_data.var( axis = 0 ) )

            min_loss = np.array( [ 10 ] )
            best_iter = 0
            best_coef = weight_vec

            avg_iter_loss = []

            # Loop for each of the max iterations
            for index in range( self.max_iterations ):
                # Calculate prediction and log loss
                pred_vec = np.matmul( learn_features, weight_vec )
                log_loss = np.ma.log( 1 + np.exp( -subtrain_labels * pred_vec ) )
                grad_loss_pred = -subtrain_labels / \
                                    ( 1 + np.exp( subtrain_labels * pred_vec ) )
                grad_loss_pred = grad_loss_pred
                grad_loss_weight_mat = grad_loss_pred * learn_features.T
                grad_vec = grad_loss_weight_mat.sum( axis = 1 )
                weight_vec -= self.step_size * grad_vec
                # get the smallest log loss
                if( not np.isinf( log_loss.mean() ) <= min_loss.mean() ):
                    min_loss = log_loss
                    best_iter = index
                    best_coef = weight_vec
                # build list of loss values
                avg_iter_loss.append( log_loss.mean() )

            # save best stuff from each pass
            results_dict[ best_iter ] = min_loss.mean()
            best_weights[ best_iter ] = best_coef
            self.avg_loss.append( avg_iter_loss )

        # get single best weight and intercept
        best_result = max( results_dict, key = results_dict.get )
        self.coef_ = best_weights[ best_result ][ 1: ]
        self.intercept_ = best_weights[ best_result ][ 0 ]

        # these get saved for plotting
        self.avg_loss = np.asarray( self.avg_loss )
        self.avg_loss = self.avg_loss.mean( axis=0 )
