class CustomerBehaviorAnalyzer:
    def missing_value(self,data):
        missing_value=data.isnull().sum()
        return missing_value