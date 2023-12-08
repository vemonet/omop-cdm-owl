# Owlready2
# Copyright (C) 2021 Jean-Baptiste LAMY
# LIMICS (Laboratoire d'informatique médicale et d'ingénierie des connaissances en santé), UMR_S 1142
# University Sorbonne Paris Nord, Bobigny, France

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import csv
import types
import datetime
import operator
import functools
import re
from collections import defaultdict

import pandas as pd
from owlready2 import (
    get_ontology,
    Thing,
    AnnotationProperty,
    ObjectProperty,
    FunctionalProperty,
    ThingClass,
    Or,
    DataProperty,
    Inverse,
)
from rdflib import Graph, Literal, Namespace, URIRef, RDF, OWL

OMOP_VERSION = "6.0"
OMOP_VERSION_MAJOR = int(float(OMOP_VERSION))
# Path to the OMOP-CDM CSV specification file
# You can get the CSV file from: https://github.com/OHDSI/CommonDataModel/tree/main/inst/csv
OMOP_CDM_FIELD_CSV = f"https://raw.githubusercontent.com/OHDSI/CommonDataModel/main/inst/csv/OMOP_CDMv{OMOP_VERSION}_Field_Level.csv"
OMOP_CDM_TABLE_CSV = f"https://raw.githubusercontent.com/OHDSI/CommonDataModel/main/inst/csv/OMOP_CDMv{OMOP_VERSION}_Table_Level.csv"

# Path where the OMOP-CDM ontology file will be created
OMOP_ONTOLOGY_FILE = f"omop_cdm_v{OMOP_VERSION_MAJOR}.ttl"
OMOP_ONTOLOGY_URL = "https://w3id.org/omop/ontology/"

print(
    f"🦉 Generating OWL ontology <{OMOP_ONTOLOGY_URL}> for OMOP CDM version {OMOP_VERSION}"
)

# If true, split the ontology in several files, corresponding to the various part of the OMOP-CDM model (clinical, survey, etc)
MODULAR = False

# True to fix datetime datatype (e.g. datetime attribute has datetime datatype and not date). In doubt, keep the default value
FIX_DATETIME = True

# Set of relations that should be reversed. Use an empty set to reverse no relation. In doubt, keep the default value
REVERSE_RELATIONS = {"person_id", "note_id", "visit_detail_id", "visit_occurrence_id"}

# Sets of tables. Please keep the default values, unless importing OMOP-CDM in a version different than 6.0
VOCABULARIES_TABLES = {
    "concept",
    "vocabulary",
    "domain",
    "concept_class",
    "concept_relationship",
    "relationship",
    "concept_synonym",
    "concept_ancestor",
    "source_to_concept_map",
    "drug_strength",
}
METADATA_TABLES = {
    "cdm_source",
    "metadata",
}
CLINICAL_TABLES = {
    "person",
    "observation_period",
    "visit_occurrence",
    "visit_detail",
    "condition_occurrence",
    "drug_exposure",
    "procedure_occurrence",
    "device_exposure",
    "measurement",
    "note",
    "note_nlp",
    "death",
}
SURVEY_TABLES = {
    "survey_conduct",
    "observation",
    "specimen",
    "fact_relationship",
}
HEALTH_SYSTEM_TABLES = {
    "location",
    "location_history",
    "care_site",
    "provider",
}
ECONOMICS_TABLES = {
    "payer_plan_period",
    "cost",
}
DERIVED_TABLES = {
    "drug_era",
    "dose_era",
    "condition_era",
}
COHORT_TABLES = {
    "cohort",
    "cohort_definition",
}

# Set of tables to import. If needed, you may restrict the set to the part of OMOP-CDM you need (as in the commented example below).
TABLES = (
    VOCABULARIES_TABLES
    | METADATA_TABLES
    | CLINICAL_TABLES
    | SURVEY_TABLES
    | HEALTH_SYSTEM_TABLES
    | ECONOMICS_TABLES
    | DERIVED_TABLES
    | COHORT_TABLES
)
# TABLES = CLINICAL_TABLES | HEALTH_SYSTEM_TABLES | DERIVED_TABLES


omop_cdm = get_ontology(OMOP_ONTOLOGY_URL)
omop_cdm.metadata.label.append(
    "OWL ontology for the Observational Medical Outcomes Partnership Common Data Model (OMOP CDM)"
)
omop_cdm.metadata.comment.append(
    f"OWL ontology for the Observational Medical Outcomes Partnership Common Data Model (OMOP CDM) version {OMOP_VERSION}."
)

# OWL ontology for the Observational Medical Outcomes Partnership Common Data Model (OMOP CDM)
if MODULAR:
    omop_cdm_vocabularies = get_ontology(f"{OMOP_ONTOLOGY_URL}vocabularies/")
    omop_cdm_metadata = get_ontology(f"{OMOP_ONTOLOGY_URL}metadata.owl")
    omop_cdm_clinical = get_ontology(f"{OMOP_ONTOLOGY_URL}clinical.owl")
    omop_cdm_survey = get_ontology(f"{OMOP_ONTOLOGY_URL}survey.owl")
    omop_cdm_health_system = get_ontology(f"{OMOP_ONTOLOGY_URL}system.owl")
    omop_cdm_economics = get_ontology(f"{OMOP_ONTOLOGY_URL}economics.owl")
    omop_cdm_derived = get_ontology(f"{OMOP_ONTOLOGY_URL}derived.owl")
    omop_cdm_cohort = get_ontology(f"{OMOP_ONTOLOGY_URL}cohort.owl")
    omop_cdm.imported_ontologies = [
        omop_cdm_clinical,
        omop_cdm_derived,
        omop_cdm_health_system,
        omop_cdm_vocabularies,
        omop_cdm_metadata,
        omop_cdm_survey,
        omop_cdm_economics,
        omop_cdm_cohort,
    ]

df = pd.read_csv(OMOP_CDM_FIELD_CSV)

table_2_owl = {}
field_2_owl = {}


def get_namespace(table):
    if not MODULAR:
        return omop_cdm

    if table in VOCABULARIES_TABLES:
        onto = omop_cdm_vocabularies
    elif table in METADATA_TABLES:
        onto = omop_cdm_metadata
    elif table in CLINICAL_TABLES:
        onto = omop_cdm_clinical
    elif table in SURVEY_TABLES:
        onto = omop_cdm_survey
    elif table in HEALTH_SYSTEM_TABLES:
        onto = omop_cdm_health_system
    elif table in ECONOMICS_TABLES:
        onto = omop_cdm_economics
    elif table in DERIVED_TABLES:
        onto = omop_cdm_derived
    elif table in COHORT_TABLES:
        onto = omop_cdm_cohort
    return onto.get_namespace(omop_cdm.base_iri)


def get_prioritary_namespace(*namespaces):
    if not MODULAR:
        return omop_cdm
    return sorted(
        namespaces,
        key=lambda namespace: omop_cdm.imported_ontologies.index(namespace.ontology),
    )[-1]


def separate_words(input_string: str) -> str:
    """Separate words in column labels (e.g. RestingECG becomes Resting ECG)"""
    # Replace underscores with spaces for snake_case
    input_string = input_string.replace("_", " ")
    # Insert spaces before capital letters for CamelCase
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", input_string)


TABLES = set(TABLES)
prop_2_domain_2_range = defaultdict(dict)

FIELDS = set()
for _i, row in df.iterrows():
    field = row["cdmFieldName"]
    if field.startswith('\\"'):
        field = field[2:]
    if field.endswith('\\"'):
        field = field[:-2]
    FIELDS.add(field)


def calcule_nom_owl(s, table):
    if s.startswith(f"{table}_"):
        s = s[len(table) + 1 :]
    elif (
        table.endswith("_exposure")
        or table.endswith("_occurrence")
        or table.endswith("_era")
    ):
        if table.endswith("_exposure"):
            table_simplifiee = table.replace("_exposure", "")
        elif table.endswith("_occurrence"):
            table_simplifiee = table.replace("_occurrence", "")
        elif table.endswith("_era"):
            table_simplifiee = table.replace("_era", "")
        if s.startswith(f"{table_simplifiee}_"):
            s = s[len(table_simplifiee) + 1 :]

    if table.endswith("_nlp") and s.startswith("nlp_"):
        s = s[4:]
    if s == "drug_concept":
        s = "concept"
    if s.endswith("_concept") and not s.endswith("_as_concept"):
        s2 = s[:-8]
        if s2 not in FIELDS:
            s = s2
    return s


if MODULAR:
    namespace = omop_cdm_clinical.get_namespace(omop_cdm.base_iri)
else:
    namespace = omop_cdm
with namespace:

    class Concept(Thing):
        label = "Concept"

    class OmopCDMThing(Thing):
        label = "OMOP CDM thing"

    class Duration(OmopCDMThing):
        label = "Duration"

    class DateDuration(Duration):
        label = "Date duration"

    class DatetimeDuration(Duration):
        label = "Datetime duration"

    class Event(OmopCDMThing):
        label = "Event"

    class ClinicalElement(OmopCDMThing):
        label = "Clinical element"

    class Exposure(ClinicalElement):
        label = "Exposure"

    class Occurrence(ClinicalElement):
        label = "Occurrence"

    class Era(ClinicalElement):
        label = "Era"

    class BasePerson(OmopCDMThing):
        label = "Base person"

    class BaseVisit(Occurrence, DateDuration, DatetimeDuration):
        label = "Base visit"

    class omop_cdm_name(AnnotationProperty):
        label = "OMOP CDM name"


table_df = pd.read_csv(OMOP_CDM_TABLE_CSV)

attribute_id = 0
for nom in TABLES:
    with get_namespace(nom):
        nom_owl = "".join(mot.capitalize() for mot in nom.split("_"))
        cls = types.new_class(nom_owl, (OmopCDMThing,))
        cls.omop_cdm_name = nom
        cls.label = separate_words(nom.capitalize())
        # Add table description from OMOP Table CSV
        description = table_df.loc[
            table_df["cdmTableName"] == nom, "tableDescription"
        ].values
        if description.size > 0:
            cls.comment.en.append(description[0])

        table_2_owl[nom] = cls

        if nom.endswith("_exposure"):
            cls.is_a = [Exposure]
        elif nom.startswith("visit_"):
            cls.is_a = [BaseVisit]
        elif nom.endswith("_occurrence"):
            cls.is_a = [Occurrence]
        elif nom.endswith("_era"):
            cls.is_a = [Era]
        elif nom == "measurement":
            cls.is_a = [Occurrence]
        elif nom == "person":
            cls.is_a = [BasePerson]
        elif nom == "provider":
            cls.is_a = [BasePerson]
        elif nom == "visit_detail":
            cls.is_a = [Occurrence]
        elif nom == "note":
            cls.is_a = [ClinicalElement]
        elif nom == "observation_period":
            cls.is_a = [ClinicalElement]

ABSTRACT_CLASSES = [ClinicalElement, BaseVisit, Exposure, Occurrence, Era, BasePerson]

for (
    table,
    field,
    required,
    type,
    userGuidance,
    etlConventions,
    isPrimaryKey,
    isForeignKey,
    fkTableName,
    fkFieldName,
    fkDomain,
    fkClass,
    unique_DQ_identifiers,
) in df.itertuples(index=False):
    if table in TABLES:
        if field.startswith('\\"'):
            field = field[2:]
        if field.endswith('\\"'):
            field = field[:-2]
        description = "\n".join(
            [i for i in [userGuidance, etlConventions] if not pd.isna(i)]
        )
        type = type.upper()

        nom_owl = field
        reverse = False
        if field.endswith("_id") and (
            field != f"{table}_id"
        ):  # Clef étrangère => relation
            range = Thing
            if field.endswith("_concept_id"):
                range = Concept

            if fkTableName and not pd.isna(fkTableName):
                if (f"{fkTableName}_ID") == fkFieldName:
                    range = table_2_owl[fkTableName.lower()]
                else:
                    assert False

            else:
                mots = description.split()
                precedent = precedent2 = ""
                for mot in mots:
                    if (
                        (mot == "table")
                        or (mot == "table;")
                        or (mot == "table,")
                        or (mot == "table.")
                    ):
                        if precedent.lower() in table_2_owl:
                            range = table_2_owl[precedent.lower()]
                            break
                        elif f"{precedent2.lower()}_{precedent.lower()}" in table_2_owl:
                            range = table_2_owl[
                                f"{precedent2.lower()}_{precedent.lower()}"
                            ]
                            break
                    precedent2 = precedent
                    precedent = mot

                else:
                    if field.endswith("_id"):
                        s = field[:-3]
                        if s in table_2_owl:
                            range = table_2_owl[s]

            candidate_namepaces = [get_namespace(table)]
            if range is not Thing:
                candidate_namepaces.append(range.namespace)

            with get_prioritary_namespace(*candidate_namepaces):
                if field in REVERSE_RELATIONS:
                    reverse = True
                    nom_owl = f"has_{table}"
                    prop = types.new_class(nom_owl, (ObjectProperty,))
                    if prop.name == "note_nlp":
                        prop.python_name = "notes_nlp"
                    else:
                        prop.python_name = prop.name[4:] + "s"
                else:
                    nom_owl = field[:-3]
                    nom_owl = calcule_nom_owl(nom_owl, table)
                    nom_owl = f"has_{nom_owl}"
                    prop = types.new_class(
                        nom_owl,
                        (
                            ObjectProperty,
                            FunctionalProperty,
                        ),
                    )
                    prop.python_name = prop.name[4:]

        else:
            nom_owl = calcule_nom_owl(nom_owl, table)
            if type.startswith("INTEGER"):
                range = int
            elif type.startswith("BIGINT"):
                range = int
            elif type.startswith("STRING"):
                range = str
            elif type.startswith("VARCHAR"):
                range = str
            elif type.startswith("NVARCHAR"):
                range = str
            elif type.startswith("CLOB"):
                range = str
            elif type.startswith("FLOAT"):
                range = float
            elif type.startswith("DATETIME"):
                if FIX_DATETIME and field.endswith("_date"):
                    nom_owl.replace("_date", "_datetime")
                range = datetime.datetime
            elif type.startswith("DATE"):
                if FIX_DATETIME and field.endswith("_datetime"):
                    nom_owl.replace("_datetime", "_date")
                range = datetime.date
            else:
                raise ValueError(f"Unknown type {type}!")

            with get_namespace(table):
                prop = types.new_class(
                    nom_owl,
                    (
                        DataProperty,
                        FunctionalProperty,
                    ),
                )

        domain = table_2_owl[table]
        if prop.name == "id":
            domain = OmopCDMThing
            range = Or([int, str])

        if isinstance(range, Or):
            range0 = "int|str"
        elif isinstance(range, ThingClass):
            range0 = range.name
        else:
            range0 = range.__name__
        if reverse:
            domain, range = range, domain
        prop_2_domain_2_range[prop][domain] = (range, required, reverse)

        if prop.name == "start_datetime":
            domain.is_a.append(DatetimeDuration)
            if OmopCDMThing in domain.is_a:
                domain.is_a.remove(OmopCDMThing)
        elif prop.name == "start_date":
            domain.is_a.append(DateDuration)
            if OmopCDMThing in domain.is_a:
                domain.is_a.remove(OmopCDMThing)
        elif prop.name == "datetime":
            domain.is_a.append(Event)
            if OmopCDMThing in domain.is_a:
                domain.is_a.remove(OmopCDMThing)

        attribute_id += 1
        prop.omop_cdm_name.append(
            f"{table_2_owl[table].omop_cdm_name.first()}.{field}#{attribute_id} AS {range0}"
        )
        prop.label = separate_words(nom_owl.capitalize())
        if reverse:
            reversed_note = "reversed relation, "
        else:
            reversed_note = ""
        if description:
            prop.comment.en.append(
                f"{reversed_note}For {separate_words(table_2_owl[table].name)}: {description}"
            )

        field_2_owl[field] = prop


ABSTRACT_CLASSES_2_CLASSES = {
    abstract_class: {
        leaf_class
        for leaf_class in abstract_class.descendants(include_self=False)
        if leaf_class.omop_cdm_name
    }
    for abstract_class in ABSTRACT_CLASSES
}

for abstract_class, leaf_classes in ABSTRACT_CLASSES_2_CLASSES.items():
    common_super_classes = functools.reduce(
        operator.and_, [set(leaf_class.is_a) for leaf_class in leaf_classes]
    )
    common_super_classes.discard(abstract_class)
    if common_super_classes:
        abstract_class.is_a.extend(common_super_classes)
        for leaf_class in leaf_classes:
            for common_super_class in common_super_classes:
                leaf_class.is_a.remove(common_super_class)


for prop, domain_2_range in prop_2_domain_2_range.items():
    if (prop.name == "start_datetime") or (prop.name == "end_datetime"):
        prop.domain.reinit([DatetimeDuration])
        prop.range.reinit([datetime.datetime])
        DatetimeDuration.is_a.append(prop.only(datetime.datetime))
    elif (prop.name == "start_date") or (prop.name == "end_date"):
        prop.domain.reinit([DateDuration])
        prop.range.reinit([datetime.date])
        DateDuration.is_a.append(prop.only(datetime.date))
    elif (prop.name == "datetime") or (prop.name == "date"):
        prop.domain.reinit([Event])
        if prop.name == "datetime":
            prop.range.reinit([datetime.datetime])
            Event.is_a.append(prop.only(datetime.datetime))
        else:
            prop.range.reinit([datetime.date])
            Event.is_a.append(prop.only(datetime.date))

    else:
        domains = set(domain_2_range.keys())
        if len(domains) > 1:
            for abstract_class, leaf_classes in ABSTRACT_CLASSES_2_CLASSES.items():
                if domains.issuperset(leaf_classes):
                    abstract_class_ranges = {
                        domain_2_range[domain] for domain in leaf_classes
                    }
                    if len(abstract_class_ranges) == 1:
                        domains.difference_update(leaf_classes)
                        domains.add(abstract_class)
                        for leaf_class in leaf_classes:
                            del domain_2_range[leaf_class]
                        domain_2_range[abstract_class] = list(abstract_class_ranges)[0]

        if len(domains) == 1:
            prop.domain.reinit([list(domains)[0]])
        elif len(domains) > 1:
            prop.domain.reinit([Or(list(domains))])

        for domain, (range, required, reverse) in domain_2_range.items():
            if domain is Thing:
                continue
            with get_prioritary_namespace(prop.namespace, domain.namespace):
                domain.is_a.append(prop.only(range))
                if required == "Yes":
                    if reverse:
                        range.is_a.append(Inverse(prop).some(domain))
                    else:
                        domain.is_a.append(prop.some(range))

        # ranges = set(domain_2_range.values())
        ranges = set(range for (range, required, reverse) in domain_2_range.values())
        if Thing in ranges:
            pass
        elif len(ranges) == 1:
            range = list(ranges)[0]
            prop.range.reinit([range])


if len(omop_cdm.Concept.is_a) > 1:
    omop_cdm.Concept.is_a.remove(Thing)

d = {}
for prop in omop_cdm.properties():
    n = prop.python_name or prop.name
    if n in d:
        print("Prop name clash for:", d[n], prop)
    else:
        d[n] = prop


# omop_cdm.save(OMOP_ONTOLOGY_FILE)
omop_cdm.save(OMOP_ONTOLOGY_FILE, format="ntriples")

g = Graph()
g.parse(OMOP_ONTOLOGY_FILE)
g.bind("omop", Namespace(OMOP_ONTOLOGY_URL))
g.bind("vann", Namespace("http://purl.org/vocab/vann/"))
g.bind(
    "owlready",
    Namespace(
        "http://www.lesfleursdunormal.fr/static/_downloads/owlready_ontology.owl#"
    ),
)
for onto_subj in g.subjects(predicate=RDF.type, object=OWL.Ontology):
    g.add(
        (
            onto_subj,
            URIRef("http://purl.org/dc/terms/license"),
            URIRef("https://www.gnu.org/licenses/lgpl-3.0"),
        )
    )
    g.add(
        (
            onto_subj,
            URIRef("http://purl.org/vocab/vann/preferredNamespacePrefix"),
            Literal("omop"),
        )
    )
    g.add(
        (
            onto_subj,
            URIRef("http://purl.org/vocab/vann/preferredNamespaceUri"),
            Literal(OMOP_ONTOLOGY_URL),
        )
    )
g.serialize(OMOP_ONTOLOGY_FILE, format="ttl")

# NOTE: to add dcterms:description:
# dcterms_onto = default_world.get_ontology("http://purl.org/dc/terms/").load()
# dcterms = dcterms_onto.get_namespace("http://purl.org/dc/terms/")
# dcterms.description[cls] = description

if MODULAR:
    omop_cdm_vocabularies.save(OMOP_ONTOLOGY_FILE.replace(".owl", "_vocabularies.owl"))
    omop_cdm_metadata.save(OMOP_ONTOLOGY_FILE.replace(".owl", "_metadata.owl"))
    omop_cdm_clinical.save(OMOP_ONTOLOGY_FILE.replace(".owl", "_clinical.owl"))
    omop_cdm_survey.save(OMOP_ONTOLOGY_FILE.replace(".owl", "_survey.owl"))
    omop_cdm_health_system.save(
        OMOP_ONTOLOGY_FILE.replace(".owl", "_health_system.owl")
    )
    omop_cdm_economics.save(OMOP_ONTOLOGY_FILE.replace(".owl", "_economics.owl"))
    omop_cdm_derived.save(OMOP_ONTOLOGY_FILE.replace(".owl", "_derived.owl"))
    omop_cdm_cohort.save(OMOP_ONTOLOGY_FILE.replace(".owl", "_cohort.owl"))

print(f"✔ OWL ontology generated successfully with {len(omop_cdm.graph)} RDF triples")
