from outputfiles.plot import plotHisto
import mysrc.constants as cst

df = plotHisto(cst.root_dir / "raw_data" / 'Algeria_ASAP_2d_data.csv',
               'Bouira', 'NDVI')
df = plotHisto(cst.root_dir / "raw_data" / r'histo_data_from_db_query.csv',
    'Bouira', 'rad')
df = plotHisto(cst.root_dir / "raw_data" / r'histo_data_from_db_query.csv',
    'Bouira', 'rainfall')
df = plotHisto(cst.root_dir / "raw_data" / r'histo_data_from_db_query.csv',
    'Bouira', 'temperature')