using PkgTemplates

# Create a new package named CompositionalJulia

t = Template(;
    user="AubreyCrane",
    authors=["A. William Crane"],
    julia=v"1.11",
    plugins=[
        Git(),
        GitHubActions(),         # For CI
        Codecov(),               # Optional: test coverage
        Documenter{GitHubActions}(), # Doc deployment
        Tests(),                 # Sets up a `test/runtests.jl`
        Readme(),                # Generates a starter README
        License(name="MIT"),
    ]
)

t("CompositionalJulia")
