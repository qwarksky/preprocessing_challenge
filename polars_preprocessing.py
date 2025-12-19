import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full", sql_output="native")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from sklearn.preprocessing import MinMaxScaler,StandardScaler, RobustScaler,Normalizer,MaxAbsScaler
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
    from functools import wraps
    return (
        MaxAbsScaler,
        MinMaxScaler,
        OneHotEncoder,
        OrdinalEncoder,
        RobustScaler,
        StandardScaler,
        mo,
        pl,
        wraps,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Loading DataSet
    """)
    return


@app.cell
def _(pl):
    pf = pl.read_parquet("datasets/taxis_str.parquet")
    pf.describe()
    return (pf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Detect which encoder Ordinal or OneHot
    * Try feature.n_unique < 10 => OneHot
    * Try feature.n_unique > 10 => Ordinal

    > Warning : it depend nature of the feature, be carefull
    """)
    return


@app.cell
def _(pf, pl):
    pf.select(pl.all().n_unique())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Preprocessing
    * Casting : to define best type feature
    * Encoders : Onehot, Ordinal
    * Scalers : MinMax, MaxAbs, Robust, Standard
    * Drop non numerical features
    """)
    return


@app.cell
def _(pl, wraps):
    def logshow(func):
        """
            decorator for function transformation
            using wraps decorator to get args and kwargs from function
            out is displaying function name, dataframe shape, columns name and dtypes
        """
        @wraps(func)
        def wrapper(*args,**kwargs):

            ret = func(*args,**kwargs)
            print(func.__name__,ret.shape,dict(zip(ret.columns,ret.dtypes)))
            return ret
        return wrapper


    @logshow
    def casting(_df:pl.DataFrame)->pl.DataFrame:
        """
            Features typing
            Parameters : _df = DataFrame polars
            Returns : DataFrame polars with correct type 
        """
        return(_df
            .cast({'passengers':pl.Int8})
            .with_columns(
                pl.col(['distance','fare','tip','tolls','total']).cast(pl.Float64),
                pl.col(['pickup','dropoff']).str.slice(0,23).str.strptime(pl.Datetime(time_unit='ms'),"%Y-%m-%d %H:%M:%S.%3f")
            )   
        )

    @logshow
    def charsinfo(_df:pl.DataFrame)->pl.DataFrame:
        """
            More details about feteatures dtypes string before casting function.
            Parameters : _df = DataFrame polars
            Returns : DataFrame polars with correct type 
        """
        return (pl.DataFrame({
                'DTYPES':_df.dtypes,
                'FEATURES':_df.columns,
                'COUNT':_df.select(pl.all().count()).transpose(),
                'NUNIQUES':_df.select(pl.all().n_unique()).transpose(),
                'NULLS':_df.select(pl.all().null_count()).transpose(),
                'TOP':_df.max().transpose(),
                'BOTTOM':_df.min().transpose(),
                'WORDS':_df.select(pl.all().cast(pl.String).str.split(' ').list.len().sum()).transpose(),
                'WORDS_min':_df.select(pl.all().cast(pl.String).str.split(' ').list.len().min()).transpose(),
                'WORDS_mean':_df.select(pl.all().cast(pl.String).str.split(' ').list.len().mean()).transpose(),
                'WORDS_max':_df.select(pl.all().cast(pl.String).str.split(' ').list.len().max()).transpose(),
                'WORDS_median':_df.select(pl.all().cast(pl.String).str.split(' ').list.len().median()).transpose(),
                'LEN_max':_df.select(pl.all().cast(pl.String).str.len_chars().max()).transpose(),
                'LEN_min':_df.select(pl.all().cast(pl.String).str.len_chars().min()).transpose(),
                'LEN_mean':_df.select(pl.all().cast(pl.String).str.len_chars().mean()).transpose(),
                'LEN_median':_df.select(pl.all().cast(pl.String).str.len_chars().median()).transpose(),
            }).sort(by='NUNIQUES',descending=False)
        )

    @logshow
    def select_by_nunique(_df:pl.DataFrame,_condition:str,_threshold:int)->pl.DataFrame:
        """
            Selection columns names diffrent condition threshold
            Parameters : _df = DataFrame polars, _threshold = nunique value
            Returns : DataFrame polars with correct type 
        """
        match _condition:
            case '<':
                return(_df.select(_df.select(pl.all().n_unique()).unpivot(index=None,on=_df.columns).filter(pl.col('value') < _threshold).to_series(0).to_list()))
            case '<=':
                return(_df.select(_df.select(pl.all().n_unique()).unpivot(index=None,on=_df.columns).filter(pl.col('value') <= _threshold).to_series(0).to_list()))
            case '>':
                return(_df.select(_df.select(pl.all().n_unique()).unpivot(index=None,on=_df.columns).filter(pl.col('value') > _threshold).to_series(0).to_list()))
            case '>=':
                return(_df.select(_df.select(pl.all().n_unique()).unpivot(index=None,on=_df.columns).filter(pl.col('value') >= _threshold).to_series(0).to_list()))
            case '=':
                return(_df.select(_df.select(pl.all().n_unique()).unpivot(index=None,on=_df.columns).filter(pl.col('value') == _threshold).to_series(0).to_list()))

    @logshow    
    def exclude_by_dtypes_cols(_df:pl.DataFrame,_dtypes:list[pl.DataType])->pl.DataFrame:
        """
            Exclude columns by dtypes
            Parameters : _df = polars DataFrame, _dtypes = list polars DataType
            Returns : DataFrame polars without ecluded columns
        """
        return(_df
                  .select(pl.all().exclude(_dtypes))
              )

    ### ENCODERS

    @logshow
    def ordinal_encoder(_df:pl.DataFrame,_to_encode:str)->pl.DataFrame:
        """
            Basic encoder based from position unique values (row index)
            Parameters : _df = DataFrame polars, _to_encode = name of feature
            Returns : polars DataFrame with new colomn ordinal
        """
        return(_df
                .join(_df
                    .select(pl.col(_to_encode).unique(maintain_order=True))
                    .with_row_index(name=_to_encode+'_ordinal')
                , on=_to_encode)
        )

    @logshow
    def value_count_encoder(_df:pl.DataFrame,_to_encode:str,_descending:bool=True,_start:int=1)->pl.DataFrame:
        '''
            Based encoder from value_counts, the greatest have the biggest value
            Parameters : _df = DataFrame polars, _to_encode = name of feature, _descending = Smallest first (True) and Greatest first (False), _start = start ordering from 1 or other int
            Returns : polars DataFrame with new colomn ordinal
        '''
        return(_df
               .join(_df
                     .select(pl.col(_to_encode).value_counts()).unnest(_to_encode)
                        .sort(by='count',descending=_descending)
                        .with_row_index(offset=_start,name=_to_encode+'_vc_ordinal')
                     , on = _to_encode)
               .drop('count')
        )



    @logshow
    def onehot_encoder(_df:pl.DataFrame,_to_encode:str,_dropfirst:bool=True)->pl.DataFrame:
        '''
            Encode each class from the feature with value 0 or 1
            Parameters : _df = DataFrame polars, _to_encode = name of feature, _dropfitst = remove 1 class to reduce bias    
            Returns : polars DataFrame with several columns for each class  
        '''
        return(_df
                .with_row_index()
                .join(_df.select(pl.col(_to_encode)).to_dummies(drop_first=_dropfirst).with_row_index(),on = 'index')
                .drop('index')
              )

    ### SCALERS 

    NUMERICS_DTYPES = pl.selectors.by_dtype([pl.Float16,pl.Float32,pl.Float64])

    @logshow
    def standard_scaler(_df:pl.DataFrame,_col:str)->pl.DataFrame:
        """
            Standard Scaler X - mean / std
            Parameters : _df = polars DataFrame, _col = feature name frome a mesure
            Returns : DataFrame polars with new _standard column
        """

        return(_df
                  .with_columns(
                      pl.col(_col)
                        .map_batches(lambda x: (x -x.mean())/x.std(ddof=1),return_dtype=pl.Float64)
                        .alias(_col+'_standard')
                  )
              )



    @logshow
    def minmax_scaler(_df:pl.DataFrame,_col:str)->pl.DataFrame:
        """
            Min Max Scaler X - min / max - min
            Parameters : _df = polars DataFrame, _col = feature name frome a mesure
            Returns : DataFrame polars with new _minmax column
        """
        return(_df
                  .with_columns(
                      pl.col(_col)
                        .map_batches(lambda x: (x - x.min())/(x.max()-x.min()),return_dtype=pl.Float64)
                      .alias(_col+'_minmax')
                  )
              )

    @logshow
    def maxabs_scaler(_df:pl.DataFrame,_col:str)->pl.DataFrame:
        """
            Max abs Scaler X / max absolute
            Parameters : _df = polars DataFrame, _col = feature name frome a mesure
            Returns : DataFrame polars with new _maxabs column
        """
        return(_df
                  .with_columns(
                      pl.col(_col)
                        .map_batches(lambda x: (x/x.abs().max()),return_dtype=pl.Float64)
                        .alias(_col+'_maxabs')
                  )
              )


    @logshow
    def robust_scaler(_df:pl.DataFrame,_col:str)->pl.DataFrame:
        """
            Robust Scaler X - median / q3 - q1
            Parameters : _df = polars DataFrame, _col = feature name frome a mesure
            Returns : DataFrame polars with new _robust column
        """
        return(_df
                 .with_columns(
                      pl.col(_col).map_batches(lambda x: (x-x.median())/(x.quantile(.75)-x.quantile(.25)),return_dtype=pl.Float64)
                        .alias(_col+'_robust')
                 )
              )
    return (
        NUMERICS_DTYPES,
        casting,
        exclude_by_dtypes_cols,
        onehot_encoder,
        select_by_nunique,
        standard_scaler,
        value_count_encoder,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Piepline
    """)
    return


@app.cell
def _(
    casting,
    exclude_by_dtypes_cols,
    onehot_encoder,
    pf,
    pl,
    standard_scaler,
    value_count_encoder,
):

    (pf
        .pipe(casting)
        .pipe(onehot_encoder,'color')
        .pipe(onehot_encoder,'payment')
        .pipe(value_count_encoder,'pickup_borough')
        .pipe(standard_scaler,'distance')
        .pipe(standard_scaler,'fare')
        .pipe(exclude_by_dtypes_cols,[pl.Datetime,pl.String])
        #.pipe(charsinfo)
        .describe()
     )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Checking preprocessing with Scklearn
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ordinal Encoder
    """)
    return


@app.cell
def _(
    OrdinalEncoder,
    casting,
    exclude_by_dtypes_cols,
    pf,
    pl,
    select_by_nunique,
):
    ordinal_enc = OrdinalEncoder()
    (pl.DataFrame(ordinal_enc.fit_transform(pf.pipe(casting).pipe(exclude_by_dtypes_cols,[pl.Int8,pl.Datetime,pl.Float64]).pipe(select_by_nunique,'>',10)),
                  schema=list(ordinal_enc.get_feature_names_out()),
                  orient='row')
        .describe()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## OneHot Encoder
    """)
    return


@app.cell
def _(
    OneHotEncoder,
    casting,
    exclude_by_dtypes_cols,
    pf,
    pl,
    select_by_nunique,
):
    onehot_enc = OneHotEncoder(drop='first',sparse_output=False)


    (pl.DataFrame(onehot_enc.fit_transform(pf.pipe(casting).pipe(exclude_by_dtypes_cols,[pl.Int8]).pipe(select_by_nunique,'<',10)),
                  schema=list(onehot_enc.get_feature_names_out()),
                  orient='row')
        .describe()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## StandardScaler
    """)
    return


@app.cell
def _(NUMERICS_DTYPES, StandardScaler, casting, pf, pl):
    sk_std_scaler = StandardScaler()
    (pl.DataFrame(data=sk_std_scaler.fit_transform(pf.pipe(casting).select([NUMERICS_DTYPES])),
                 schema=list(sk_std_scaler.get_feature_names_out()),
                 orient='row')
        .describe()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MinMaxScaler
    """)
    return


@app.cell
def _(MinMaxScaler, NUMERICS_DTYPES, casting, pf, pl):
    sk_minmax_scaler = MinMaxScaler()
    (pl.DataFrame(data=sk_minmax_scaler.fit_transform(pf.pipe(casting).select([NUMERICS_DTYPES])),
                 schema=list(sk_minmax_scaler.get_feature_names_out()),
                 orient='row')
        .describe()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## RobustScaler
    """)
    return


@app.cell
def _(NUMERICS_DTYPES, RobustScaler, casting, pf, pl):
    sk_robust_scaler = RobustScaler()
    (pl.DataFrame(data=sk_robust_scaler.fit_transform(pf.pipe(casting).select([NUMERICS_DTYPES])),
                 schema=list(sk_robust_scaler.get_feature_names_out()),
                 orient='row')
        .describe()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MaxAbs Scaler
    """)
    return


@app.cell
def _(MaxAbsScaler, NUMERICS_DTYPES, casting, pf, pl):
    sk_maxabs_scaler = MaxAbsScaler()
    (pl.DataFrame(data=sk_maxabs_scaler.fit_transform(pf.pipe(casting).select([NUMERICS_DTYPES])),
                 schema=list(sk_maxabs_scaler.get_feature_names_out()),
                 orient='row')
        .describe()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Conclusion

    * Actually with this code it's better to use Scikit-Learn preprocessing
    * Probably in feature, i will find another way to be faster with Polars
    """)
    return


if __name__ == "__main__":
    app.run()
