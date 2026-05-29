import pandas as pd

def format_dataframe(xlsx_path: str = 'data/OMPP_master_dataframe_filtered.xlsx'):
    """ Format the raw excel file and calculate mean values for each assay type, then save to a new csv file """
    path = xlsx_path
    df = pd.read_excel(path, sheet_name='OMPP_master_COLORED')

    out_paths = ["data/OMPP_master_long.csv", "data/OMPP_master_short.csv"]

    # We have two different datasets one well characterized and one less so
    # Well characterized: rows 1-15
    # Less characterized: rows 16-34

    for i, output_csv_path in enumerate(out_paths):
        if i == 0:
            df = df.iloc[0:33, :] # Select rows for long
        else:
            df = df.iloc[0:15, :] # Select rows for short
        #print(df.columns)

        #print(df.head())

        # Drop columns by index
        potentiation_assay_idx = [i+df.columns.get_loc("Potentiation") for i in range(3)] #[11, 12, 13]
        mic_assay_idx = [i+df.columns.get_loc("MIC") for i in range(3)] #[15, 16, 17]
        npn_assay_idx = [i+df.columns.get_loc("NPN Endpoint") for i in range(3)] #[18, 19, 20]
        disc_assay_idx = [df.columns.get_loc("Disc3(5)"), df.columns.get_loc("Disc3(5).1"), df.columns.get_loc("Disc3(5).2")] #[21, 23, 25]
        cytotoxicity_assay_idx = [i+df.columns.get_loc("Cytotox") for i in range(3)] #[27, 28, 29]
        hemolysis_assay_idx = [i+df.columns.get_loc("Hemolysis % 200uM") for i in range(2)] #[30, 31]

        # Calculate mean values for each assay type
        potentiation_mean = df.iloc[1:, potentiation_assay_idx].mean(axis=1)
        npn_mean = df.iloc[1:, npn_assay_idx].mean(axis=1)
        disc_mean = df.iloc[1:, disc_assay_idx].mean(axis=1)
        cytotoxicity_mean = df.iloc[1:, cytotoxicity_assay_idx].mean(axis=1)
        hemolysis_mean = df.iloc[1:, hemolysis_assay_idx].mean(axis=1)

        # Handle MIC separately due to possible '>' values
        mic_values = (
            df.iloc[1:, mic_assay_idx]
            .astype(str)
            .apply(lambda col: col.str.lstrip(">"))
            .apply(pd.to_numeric, errors="coerce")
        )
        mic_mean = mic_values.mean(axis=1)

        # Get the sequence columns
        clean_df = pd.DataFrame()
        clean_df['OMPP nr'] = df.iloc[1:, 0]
        clean_df['Peptide'] = df.iloc[1:, 1]
        clean_df['Sequence'] = df.iloc[1:, 3]

        # Add the mean values as new columns
        clean_df['Potentiation_Mean'] = potentiation_mean
        clean_df['MIC_Mean'] = mic_mean
        clean_df['NPN_Mean'] = npn_mean
        clean_df['DISC_Mean'] = disc_mean
        clean_df['Cytotoxicity_Mean'] = cytotoxicity_mean
        clean_df['Hemolysis_Mean'] = hemolysis_mean

        # Display the dataframe
        # print(clean_df)

        # Save the cleaned dataframe to a new csv file
        clean_df.to_csv(output_csv_path, index=False)

def format_extended_dataframe(xlsx_path: str = 'data/new_OMPPs_dataframe.xlsx'):
    """ Format the raw excel file and calculate mean values for each assay type, then save to a new csv file """
    path = xlsx_path
    df = pd.read_excel(path, sheet_name='OMPP_master_COLORED')

    output_csv_path = "data/OMPP_new_master.csv"

    #print(df.head())

    # Drop columns by index
    potentiation_assay_idx = [i+df.columns.get_loc("Potentiation") for i in range(3)] #[11, 12, 13]
    mic_assay_idx = [i+df.columns.get_loc("MIC") for i in range(3)] #[15, 16, 17]
    npn_assay_idx = [i+df.columns.get_loc("NPN Endpoint") for i in range(3)] #[18, 19, 20]
    disc_assay_idx = [df.columns.get_loc("Disc3(5)"), df.columns.get_loc("Disc3(5).1"), df.columns.get_loc("Disc3(5).2")] #[21, 23, 25]
    cytotoxicity_assay_idx = [i+df.columns.get_loc("Cytotox") for i in range(3)] #[27, 28, 29]
    hemolysis_assay_idx = [i+df.columns.get_loc("Hemolysis % 200uM") for i in range(2)] #[30, 31]

    # Calculate mean values for each assay type
    potentiation_mean = df.iloc[1:, potentiation_assay_idx].mean(axis=1)
    npn_mean = df.iloc[1:, npn_assay_idx].mean(axis=1)
    disc_mean = df.iloc[1:, disc_assay_idx].mean(axis=1)
    cytotoxicity_mean = df.iloc[1:, cytotoxicity_assay_idx].mean(axis=1)
    hemolysis_mean = df.iloc[1:, hemolysis_assay_idx].mean(axis=1)

    # Handle MIC separately due to possible '>' values
    mic_values = (
        df.iloc[1:, mic_assay_idx]
        .astype(str)
        .apply(lambda col: col.str.lstrip(">"))
        .apply(pd.to_numeric, errors="coerce")
    )
    mic_mean = mic_values.mean(axis=1)

    # Get the sequence columns
    clean_df = pd.DataFrame()
    clean_df['OMPP nr'] = df.iloc[1:, 0]
    clean_df['Peptide'] = df.iloc[1:, 1]
    clean_df['Sequence'] = df.iloc[1:, 3].str.strip()

    # Add the mean values as new columns
    clean_df['Potentiation_Mean'] = potentiation_mean
    clean_df['MIC_Mean'] = mic_mean
    clean_df['NPN_Mean'] = npn_mean
    clean_df['DISC_Mean'] = disc_mean
    clean_df['Cytotoxicity_Mean'] = cytotoxicity_mean
    clean_df['Hemolysis_Mean'] = hemolysis_mean

    # Select only rows where MIC_Mean is not NaN
    clean_df = clean_df[clean_df['MIC_Mean'].notna()]


    # Save the cleaned dataframe to a new csv file
    clean_df.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    format_dataframe()
    format_extended_dataframe()