# 🦉OWL ontology for the OMOP CDM 🩺

[![Update documentation website](https://github.com/vemonet/omop-cdm-owl/actions/workflows/publish.yml/badge.svg)](https://github.com/vemonet/omop-cdm-owl/actions/workflows/publish.yml)

A repository to build an OWL ontology for the [Observational Medical Outcomes Partnership Common Data Model](https://www.ohdsi.org/data-standardization/) (OMOP CDM).

A documentation website is built and published at **[vemonet.github.io/omop-cdm-owl](https://vemonet.github.io/omop-cdm-owl)**

This project reuse and modify the script published in this publication: https://hal.science/hal-03479322/document. The original script can be found [here](https://bitbucket.org/jibalamy/owlready2/src/master/pymedtermino2/omop_cdm/import_omop_cdm.py)

The script to build the ontology has been modified to add:

* Labels for classes and properties
* Change the ontology URI to https://w3id.org/omop/ontology/
* Ontology metadata (license, label, description, preferred prefix and namespace)

♻️ The documentation website hosted at [vemonet.github.io/omop-cdm-owl](https://vemonet.github.io/omop-cdm-owl) is automatically updated by a GitHub Action at every change to the ontology file.

## 📥 Install dependencies

<details><summary>Make sure Java ~17 and python >=3.8 are installed. We recommend to enable a python virtual environment.</summary>

Create the virtual environment:
```bash
python -m venv .venv
```

Activate the virtual environment:
```bash
source .venv/bin/activate
```
</details>

```bash
./scripts/install.sh
```

## 🦉 Generate the OWL ontology

Run the script:

```bash
python generate_omop_owl.py
```


## 📖 Generate the docs locally

Generate ontology and build the docs website:

```bash
./scripts/build.sh
```

Start a web server to check the generated docs locally:

```bash
./scripts/start.sh
```
