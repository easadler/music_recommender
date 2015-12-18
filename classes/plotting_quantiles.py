class QuantileEDA(object):
    def __init__(self, df, by_col, value_col):
        self.by_col = by_col
        self.value_col = value_col

        self.metrics  = ['sum', 'mean', 'count', 'max', 'min']
        self.df = self.compute_quantiles(df)
        grouped = df.groupby(self.by_col)
        self.df_g = grouped.aggregate(lambda x: list(x)).reset_index()


    def compute_quantiles(self, df):
        f = {self.value_col: self.metrics}
        return df_t.groupby('userid').agg(f).reset_index()

    def plot_quantiles(self, metric):
        f, ax = plt.subplots(2, 2,sharex=True, sharey=True)

        coords = [[0,0],[0,1],[1,0],[1,1]]
        for ind, i in enumerate([[0,0.25],[0.25,0.5],[0.5,0.75],[0.75,1]]):
            df_q = self.df.ix[(self.df[self.value_col][metric] < self.df[self.value_col][metric].quantile(i[1])) & \
                              (self.df[self.value_col][metric] > self.df[self.value_col][metric].quantile(i[0])),:]
            percentile = set(df_q.userid)
            ax[coords[ind][0], coords[ind][1]].set_title('quantile ' + str(ind + 1))
            for i in xrange(len(self.df_g)):
                x = self.df_g.ix[i,:]
                if x['userid'] in percentile:
                    ax[coords[ind][0], coords[ind][1]].plot(range(len(x['plays'])), sorted(x['plays']))

        f.suptitle('Smallest to Largest Listens by User: quantiles by ' + metric)
        plt.show()
        print self.df[self.value_col][metric].describe().T