import pandas as pd
import DBUtil
from xlrd import XLRDError
import numpy as np
from collections.abc import Iterable

class LengthMatcher():
    
    def __init__(self, mastersheet = "final"):
        self.read_sheets = []
        self.mastersheet = mastersheet
        pass
        
        
    def read(self, masterfile, years, months, cleanData = True, tagEquivalences = None, verbose = False):
        self.masterfile = masterfile
        #Create list for quick access of possible alternate tags
        self.tagEquivalences = {}
        if tagEquivalences is not None:
            for k, v in tagEquivalences.items():
                if v in self.tagEquivalences:
                    self.tagEquivalences[v].append(k)
                else:
                    self.tagEquivalences[v] = [k]
                if k in self.tagEquivalences:
                    self.tagEquivalences[k].append(v)
                else:
                    self.tagEquivalences[k] = [v]
        dfs = None
        try:
            dfs = [_preprocess_df(self.masterfile, self.mastersheet)]
            if verbose:
                print("LengthMatcher: Read sheet %s" % self.mastersheet)
            self.read_sheets.append("final")
        except XLRDError as e:
            if verbose:
                print("LengthMatcher: Failed to retrieve data for date %s: %s" % (date, str(e)))
        
        for date, year, monthIdx in DBUtil.generate_dates(years=years, months=months, order= DBUtil.DateOrder.MONTH_YEAR):
            try:
                df = _preprocess_df(self.masterfile, date)
                dfs.append(df)
                if verbose:
                    print("LengthMatcher: Read sheet %s" % date)
                self.read_sheets.append(date)
            except XLRDError as e:
                if verbose:
                    print("LengthMatcher: Failed to retrieve data for date %s: %s" % (date, str(e)))
        self.joint_df = pd.concat(dfs, axis = 0)
        self.joint_df["TAG"] = self.joint_df["TAG"].astype(str)
        self.joint_df["Fish ID"] = self.joint_df["Fish ID"].astype(str)
        if cleanData:
            self.joint_df = _clean(self.joint_df)
        
    def getLength(self, cave, tagID, year, month):
        tag = _standardise_tag(tagID)
        df = self.joint_df[(self.joint_df["Location"] == "C%d" % cave) & 
                             (self.joint_df["Fish ID"] == tag) | (self.joint_df["TAG"] == tag) &
                             (self.joint_df["year"] == year) &
                             (self.joint_df["season"] == month)]
        if (df["length"].size > 0):
            return df["length"].iloc[0]
        else:
            if tagID in self.tagEquivalences:
                for tag in self.tagEquivalences[tagID]:
                    tag = _standardise_tag(tag)
                    df = self.joint_df[(self.joint_df["Location"] == "C%d" % cave) & 
                                         (self.joint_df["Fish ID"] == tag) | (self.joint_df["TAG"] == tag) &
                                         (self.joint_df["year"] == year) &
                                         (self.joint_df["season"] == month)]
                    if (df["length"].size > 0):
                        return df["length"].iloc[0]
                    else:
                        return None
            else:
                return None
        
        
        
    def getLengthDistribution(self, cave = None, tag = None, year = None, month = None):
        df = self.joint_df
        if cave is not None:
            df = df[df["Location"] == "C%d" % cave]
        if year is not None:
            df = df[df["year"] == year]
        if month is not None:
            df = df[df["season"] == month]
        if tag is not None:
            if isinstance(tag, Iterable) and not isinstance(tag, str):
                tags = [_standardise_tag(t) for t in tag]
                df = df[(df["Fish ID"].isin(tags)) | (df["TAG"].isin(tags))]
            else:
                tag = _standardise_tag(tag)
                df = df[(df["Fish ID"] == tag) | (df["TAG"] == tag)]
        if (df["length"].size > 0):
            return df["length"]
        else:
            return None
        
def _standardise_tag(tagID):
    tagID = str(tagID)
    if "CAL" in tagID:
        if tagID != "-":  #Is in diffrent format in excel sheet than in pictures. Convert.
            tagID = tagID[:3] + "-" + tagID[3:5] + tagID[6:]
    elif "c:" == tagID[:2]:
        tagID = tagID[2:]
    return tagID
        
def _preprocess_df(path, sheet):
    df = None
    try:
        df = pd.read_excel(path, sheet, dtype={"TAG":str,'Fish ID':str}, parse_dates = ['Date of capture'])
    except ValueError:
        df = pd.read_excel(path, sheet, dtype={"TAG":str,'Fish ID':str})
    df.drop(columns = [col for col in df.columns if "Unnamed" in col], inplace = True)
    df.rename(_columnStandardiser, axis = 1, inplace = True)
    return df
        
def _columnStandardiser(name):
    if "comment" in str(name).lower() or "note" in str(name).lower():
        return "comment"
    elif "weight" in str(name).lower():
        return "weight"
    elif "length" in str(name).lower():
        return "length"
    elif "tag" in str(name).lower():
        return "TAG"
    elif "cave" in str(name).lower():
        return "Location"
    else:
        return str(name).strip()

    
def _clean(joint_df):
    #Keep only records with some form of ID
    joint_df = joint_df[joint_df["Fish ID"].notna() | joint_df["TAG"].notna()]
    #Some 0 are cut off because they were inputted using some weird russian format. Luckily 5 digit IDs do not exist. So just find all those and prepend a 0
    joint_df["TAG"].mask((joint_df["TAG"].str.len() == 5), lambda tag: "0" + tag, inplace = True)
    #Drop records with no season
    #N.B. Aug 2016 cave 27 has entries with no season or year taken in July
    joint_df = joint_df[joint_df["season"].notnull()]
    joint_df['season'] = joint_df['season'].apply(lambda x: x.lower().strip())
    #Fix date of capture for august 2018
    joint_df.loc[joint_df['Date of capture'].dt.year == 2022, 'Date of capture'] -= pd.to_timedelta(4 * 365.25, unit='D')
    
    #Fix locations
    def locationFormatter(loc):
        return "C%d" % loc  if isinstance(loc, int) else loc.upper()

    joint_df = joint_df[joint_df["Location"].notnull()]
    joint_df["Location"] = joint_df["Location"].apply(locationFormatter)
    
    #Fix capture methods
    joint_df["Capture method"] = joint_df["Capture method"].apply(lambda x: ("electrofishing" if x == "electro" else x.lower()) if isinstance(x, str) else x)
    
    #Remove all lengths which contain strings. I know this syntax is weird. But its simple as the .str will only access str objects if there are any
    try:
        joint_df.loc[joint_df["length"].str.contains(".*").notnull(), "length"] = None
    except AttributeError:
        pass # Only thrown if there is no strings
    #Then convert to float
    joint_df["length"] = joint_df["length"].astype(np.float)
    #According to R code all values below 25 are typos and need to be multiplied by 10
    joint_df.loc[joint_df["length"] < 25, "length"] *= 10
    joint_df = joint_df.drop_duplicates(subset=None, keep='first')
    
    return joint_df