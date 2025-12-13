import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full", sql_output="native")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    return (pl,)


@app.cell
def _(pl):
    pf = pl.read_parquet("datasets/taxis_str.parquet")
    pf.head()
    return (pf,)


@app.cell
def _(pl):
    def casting(_df:pl.DataFrame)->pl.DataFrame():
        return(_df
            .cast({'passengers':pl.Int8})
            .with_columns(
                pl.col(['distance','fare','tip','tolls','total']).cast(pl.Float64),
                pl.col(['pickup','dropoff']).str.slice(0,23).str.strptime(pl.Datetime(time_unit='ms'),"%Y-%m-%d %H:%M:%S.%3f")
            )   
        )
    return (casting,)


@app.cell
def _(casting, pf, pl):
    (pf.pipe(casting)
 
      .join( pf.select(
        pl.col("color").unique(maintain_order=True))
      .with_row_index(name="ordinal"), on="color"))

    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
