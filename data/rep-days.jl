# look at documentation for representative periods finder 
# --> built for older versions of Julia, dependencies cause issues, older versions of several packages are required, and the code is not compatible with the latest versioons of these packages
using JuMP
using RepresentativePeriodsFinder
using Gurobi
#=
yaml_files = [
    #"rep_days_2.yaml",
    #"rep_days_4.yaml",
    #"rep_days_8.yaml",
    #"rep_days_16.yaml",
    #"rep_days_30.yaml"
    "nl34_rep_days_8.yaml",
]
# warning - extremely long runtimes possible, especially for larger numbers of representative periods (e.g. 16 or 30), hours of runtime expected

for f in yaml_files
    find_representative_periods(f; optimizer = optimizer_with_attributes(Cbc.Optimizer))
end    

=#

find_representative_periods("nl34_rep_days_8.yaml"; optimizer = optimizer_with_attributes(Gurobi.Optimizer, "MIPGap" => 0.05, "TimeLimit" => 6600, "LogFile" => "nl34_rep_days_8.txt"))
