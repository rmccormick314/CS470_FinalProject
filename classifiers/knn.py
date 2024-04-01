class MyKNN():
    def __init__( self, n_neighbors ):
        # Initialize the class with number of desired neighbors
        # If initialized with param as list, change to int
        if ( isinstance( n_neighbors, list ) ):
            self.n_neighbors = n_neighbors[ 0 ]
        else:
            self.n_neighbors = n_neighbors
        self.train_features = []
        self.train_labels = []

    def fit( self, X, y ):
        # Stores training data
        self.train_features = X
        self.train_labels = y

    def predict( self, test_features ):
        # Create array to hold prediction
        predicted_labels = []

        # Loop over data entries
        for test_index in range( len( test_features ) ):
            # Create array for holding best neighbors for this entry
            nearest_labels = []

            # Convert n_neighbors to int if set as a list
            if ( isinstance( self.n_neighbors, list ) ): #My_CV will set as list
                self.n_neighbors = self.n_neighbors[ 0 ]

            # Calculate best neighbors using code from class
            test_i_features = test_features[ test_index,: ]
            diff_mat = self.train_features - test_i_features
            squared_diff_mat = diff_mat ** 2
            squared_diff_mat.sum( axis = 0 ) # sum over columns, for each row.
            distance_vec = squared_diff_mat.sum( axis = 1 ) # sum over rows
            sorted_indices = distance_vec.argsort()
            nearest_indices = sorted_indices[ :self.n_neighbors ]

            # Get the output values for selected neighbors
            for entry_index in nearest_indices:
                nearest_labels.append( self.train_labels[ entry_index ] )

            # Add this entry's predicted outcome to the list
            predicted_labels.append( mode( nearest_labels ) )

        # Return list of predicted outcomes
        return( predicted_labels )
