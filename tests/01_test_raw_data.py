import unittest
import os
import dotenv
import tifffile
import pandas as pd


class TestRawData(unittest.TestCase):
    def setUp(self):
        project_dir = os.path.join(os.getcwd(), os.pardir)
        dotenv_path = os.path.join(project_dir, '.env')
        found = dotenv.load_dotenv(dotenv_path)
        self.found = found

    def test_environ(self):
        assert self.found

    def test_load_dem(self):
        mola = tifffile.imread(os.getenv("DM_MarsDEM"))

    def test_load_robbins(self):
        df = pd.read_table(os.getenv("DM_CraterTable"),
                           sep='\t',
                           engine='python')
       

if __name__ == '__main__':
    unittest.main()