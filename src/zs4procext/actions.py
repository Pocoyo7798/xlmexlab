from typing import Any, ClassVar, Dict, List, Optional

from pydantic import BaseModel, validator, PrivateAttr
import re

from zs4procext.parser import (
    Conditions,
    ComplexParametersParser,
    DimensionlessParser,
    KeywordSearching,
    ParametersParser,
    SchemaParser,
)

class Chemical(BaseModel):
    name: str = ""
    quantity: Optional[List[str]] = []
    concentration: Optional[List[str]] = []
    _chemical_type: str = PrivateAttr(default="reactant")
    
    def get_chemical(self, schema: str, schema_parser: SchemaParser, complex_parser: ComplexParametersParser = None) -> bool:
        """get the chemical name from a schema

        Args:
            schema: string containing the schema

        Returns:
            True if the chemical was added dropwise or Flase otherwise
        """
        chemical_list = schema_parser.get_atribute_value(schema, "name")
        dropwise_list = schema_parser.get_atribute_value(schema, "dropwise")
        if len(chemical_list) == 0:
            pass
        elif len(chemical_list) == 1:
            self.name = chemical_list[0]
        else:
            print("Warning: Two different chemical names have been found!")
            self.name = chemical_list[0]
        if len(dropwise_list) == 0:
            dropwise = "False"
        elif len(dropwise_list) == 1:
            dropwise = dropwise_list[0]
        else:
            print("Warning: Two different dropwise values have been found!")
            dropwise = dropwise_list[0]
        dropwise = dropwise.strip()
        if dropwise.lower() == "true":
            new_dropwise = True
        else:
            new_dropwise = False
        if len(schema_parser.get_atribute_value(schema, "type")) > 0:
            if schema_parser.get_atribute_value(schema, "type")[0].lower() == "final solution":
                self._chemical_type = "final solution"
        concentration_list: List[str] = []
        if complex_parser is not None:
            complex_conditions = complex_parser.get_parameters(
            schema
        )
            concentration_list = complex_conditions.concentration
        if concentration_list == []:
            concentration_list: List[str] = schema_parser.get_atribute_value(schema, "concentration")
        if len(concentration_list) == 0:
            pass
        elif concentration_list[0].replace(",", "").strip().lower() == "n/a":
            pass
        elif concentration_list[0].replace(",", "").strip().lower() == "":
            pass
        else:
            self.concentration = concentration_list
        return new_dropwise
    
    def get_quantity(self, text: str, amount_parser: ParametersParser, get_concentration: bool=False) -> Any:
        """get the amount of a chemical inside a string

        Args:
            text: string to be analysed

        Raises:
            AttributeError: If the theres no ParameterParser loaded

        Returns:
            the amount of adding repetions of a chemical and concentration if asked for
        """
        amount: Conditions = amount_parser.get_parameters(text)
        amount_dict = amount.amount
        self.quantity = amount_dict["value"]  # type: ignore
        if len(amount_dict["repetitions"]) == 0:  # type: ignore
            max_repetitions: int = 1
        else:
            max_repetitions = int(max(amount_dict["repetitions"]))  # type: ignore
        if get_concentration is True:
            return amount.concentration, max_repetitions
        else:
            return max_repetitions

class ChemicalInfo(BaseModel):
    chemical_list: list[Chemical] = []
    dropwise: list[bool] = []
    final_solution: Optional[Chemical] = None
    repetitions: int = 1
    chemical_list: list[Chemical] = []
    final_solution: Optional[Chemical] = None


class Actions(BaseModel):
    action_name: str = ""
    action_context: str = ""
    type: ClassVar[Optional[str]] = None
    
    def generate_dict(self) -> Dict[str, Any]:
        action_name: str = self.action_name
        if type(self) is Grind:
            action_dict = self.model_dump(
                exclude={"action_name", "action_context", "size"}
            )
        elif type(self) in set([MakeSolution]):
            action_dict = self.model_dump(
                exclude={"action_name", "action_context", "pressure"}
            )
        else:
            action_dict = self.model_dump(
                exclude={"action_name", "action_context"}
            )
        return {"action": action_name, "content": action_dict}

class ActionsWithchemicals(Actions):
    type: ClassVar[Optional[str]] = "onlychemicals"
    
    @classmethod
    def validate_chemicals(
        cls,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        context: str,
        banned_parser: KeywordSearching,
        complex_parser: ComplexParametersParser=None,
    ) -> ChemicalInfo:
        chemical_info = ChemicalInfo()
        repetitions_list: List[int] = []
        for schema in schemas:
            new_chemical: Chemical = Chemical()
            dropwise = new_chemical.get_chemical(schema, schema_parser, complex_parser=complex_parser)
            banned_names: List[str] = banned_parser.find_keywords(new_chemical.name.lower())
            if len(schemas) > 1:
                repetitions = new_chemical.get_quantity(schema, amount_parser)
            else:
                repetitions = new_chemical.get_quantity(context, amount_parser)
            if new_chemical.name == "":
                pass
            elif new_chemical.name.strip().lower() == "n/a":
                pass
            elif len(banned_names) > 0:
                pass
            elif new_chemical._chemical_type == "final solution":
                chemical_info.final_solution = new_chemical
            else:
                chemical_info.chemical_list.append(new_chemical)
                chemical_info.dropwise.append(dropwise)
                repetitions_list.append(repetitions)
        if len(repetitions_list) == 0:
            chemical_info.repetitions = 1
        else:
            chemical_info.repetitions = max(repetitions_list)
        return chemical_info

class ActionsWithConditons(Actions):
    type: ClassVar[Optional[str]] = "onlyconditions"

    def validate_conditions(self, conditions_parser: ParametersParser, complex_conditions_parser: Optional[ComplexParametersParser]=None, add_others: bool=False) -> None:
        conditions: Dict[str, Any] = conditions_parser.get_parameters(
            self.action_context
        ).__dict__
        if complex_conditions_parser is not None:
            complex_conditions = complex_conditions_parser.get_parameters(
            self.action_context
        ).__dict__
        for atribute in self.__dict__.keys():
            try:
                if atribute == "atmosphere":
                    new_value = conditions[atribute]
                else:
                    new_value = conditions[atribute][0]
            except Exception:
                if atribute == "action_name":
                    new_value = self.__dict__[atribute]
                elif len(conditions["other"]) > 0 and add_others is True:
                    new_value = conditions["other"][0]
                else:
                    new_value = self.__dict__[atribute]
            if complex_conditions_parser is not None:
                try:
                    new_value = complex_conditions[atribute][0]
                except Exception:
                    pass
            setattr(self, atribute, new_value)


class ActionsWithChemicalAndConditions(Actions):
    type: ClassVar[Optional[str]] = "chemicalsandconditions"

    @classmethod
    def validate_chemicals(
        cls,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        context: str,
        banned_parser: KeywordSearching,
        complex_parser: ComplexParametersParser=None,
    ) -> ChemicalInfo:
        chemical_info = ChemicalInfo()
        repetitions_list: List[int] = []
        for schema in schemas:
            new_chemical: Chemical = Chemical()
            dropwise = new_chemical.get_chemical(schema, schema_parser, complex_parser=complex_parser)
            banned_names: List[str] = banned_parser.find_keywords(new_chemical.name.lower())
            if len(schemas) > 1:
                repetitions = new_chemical.get_quantity(schema, amount_parser)
            else:
                repetitions = new_chemical.get_quantity(context, amount_parser)
            if new_chemical.name == "":
                pass
            elif new_chemical.name.strip().lower() == "n/a":
                pass
            elif len(banned_names) > 0:
                pass
            elif new_chemical._chemical_type == "final solution":
                chemical_info.final_solution = new_chemical
            else:
                chemical_info.chemical_list.append(new_chemical)
                chemical_info.dropwise.append(dropwise)
                repetitions_list.append(repetitions)
        if len(repetitions_list) == 0:
            chemical_info.repetitions = 1
        else:
            chemical_info.repetitions = max(repetitions_list)
        return chemical_info

    def validate_conditions(self, conditions_parser: ParametersParser, complex_conditions_parser: Optional[ComplexParametersParser]=None, add_others: bool=False) -> None:
        conditions: Dict[str, Any] = conditions_parser.get_parameters(
            self.action_context
        ).__dict__
        if complex_conditions_parser is not None:
            complex_conditions = complex_conditions_parser.get_parameters(
            self.action_context
        ).__dict__
        for atribute in self.__dict__.keys():
            try:
                if atribute == "atmosphere":
                    new_value = conditions[atribute]
                else:
                    new_value = conditions[atribute][0]
            except Exception:
                if atribute == "action_name":
                    new_value = self.__dict__[atribute]
                elif len(conditions["other"]) > 0 and add_others is True:
                    new_value = conditions["other"][0]
                else:
                    new_value = self.__dict__[atribute]
            if complex_conditions_parser is not None:
                try:
                    new_value = complex_conditions[atribute][0]
                except Exception:
                    pass
            setattr(self, atribute, new_value)

class Treatment(ActionsWithChemicalAndConditions):
    solutions: List[Chemical] = []
    suspension_concentration: Optional[str] = None
    temperature: Optional[str] = None
    duration: Optional[str] = None
    repetitions: int = 1
    
    @classmethod
    def generate_treatment(
        cls,
        name,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        conditions_parser: ParametersParser,
        banned_parser: KeywordSearching
    ) -> List[Dict[str, Any]]:
        action: Treatment = cls(action_name=name, action_context=context)
        action.validate_conditions(conditions_parser)
        chemicals_info: ChemicalInfo = action.validate_chemicals(
            schemas, schema_parser, amount_parser, action.action_context, banned_parser
        )
        if len(chemicals_info.chemical_list) == 0:
            pass
        else:
            action.solutions = chemicals_info.chemical_list
            action.repetitions = chemicals_info.repetitions
        concentration: List[str] = re.findall(r'\d+', str(action.suspension_concentration))
        list_of_actions: List[Any] = []
        if len(action.solutions) > 0:
            list_of_actions.append(NewSolution(action_name="NewSolution").generate_dict())
            for solution in action.solutions:
                new_action: Actions = Add(action_name="Add", material=solution)
                list_of_actions.append(new_action.generate_dict())
        if action.temperature is not None:
            new_action = SetTemperature(action_name="SetTemperature", temperature=action.temperature)
            list_of_actions.append(new_action.generate_dict())
        if len(concentration) > 0:
            new_sample: Dict[str, Any] = {'action': 'Add', 'content': {'material': {'name': 'sample', 'quantity': ['1 g'], 'concentration': []}}, 'dropwise': False, 'duration': None, 'ph': None}
            list_of_actions.append(new_sample)
        if action.duration is not None:
            new_action = Stir(action_name="Stir", duration=action.duration)
            list_of_actions.append(new_action.generate_dict())
        if action.repetitions > 1:
            list_of_actions.append(Repeat(action_name="Repeat", amount=action.repetitions))
        elif action.repetitions == 1:
            list_of_actions.extend(Repeat.generate_action(context))
        list_of_actions.extend(Repeat.generate_action(context))
        return list_of_actions

### Actions for Organic Synthesis

class PH(ActionsWithChemicalAndConditions):
    ph: Optional[str] = None
    material: Optional[Chemical] = None
    dropwise: bool = False
    temperature: Optional[str] = None

    @classmethod
    def generate_action(
        cls,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        conditions_parser: ParametersParser,
        banned_parser: KeywordSearching,
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="PH", action_context=context)
        action.validate_conditions(conditions_parser)
        chemicals_info = action.validate_chemicals(
            schemas, schema_parser, amount_parser, action.action_context, banned_parser,
        )
        if len(chemicals_info.chemical_list) == 0:
            pass
        elif len(chemicals_info.chemical_list) == 1:
            action.material = chemicals_info.chemical_list[0]
        else:
            action.material = chemicals_info.chemical_list[0]
            print(
                "Warning: More than one Material have been found on Partition object, only the first one was considered"
            )
        dimensionless_values = DimensionlessParser.get_dimensionless_numbers(context)
        if len(dimensionless_values) == 0:
            pass
        elif len(dimensionless_values) == 1:
            action.ph = dimensionless_values[0]
        else:
            action.ph = dimensionless_values[0]
            print(
                "Warning: More than one dimentionless value was found for the pH, only the first one was considered"
            )
        return [action.generate_dict()]


class Add(ActionsWithChemicalAndConditions):
    material: Optional[Chemical] = None
    dropwise: bool = False
    temperature: Optional[str] = None
    atmosphere: List[str] = []
    duration: Optional[str] = None
    ph: Optional[str] = None
    
    @classmethod
    def generate_action(
        cls,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        conditions_parser: ParametersParser,
        ph_parser: KeywordSearching,
        banned_parser: KeywordSearching,
        complex_parser: ComplexParametersParser=None,
    ) -> List[Dict[str, Any]]:
        action: Add = cls(action_name="Add", action_context=context)
        action.validate_conditions(conditions_parser)
        chemicals_info: ChemicalInfo = action.validate_chemicals(
            schemas, schema_parser, amount_parser, action.action_context, banned_parser, complex_parser=complex_parser
        )
        if len(ph_parser.find_keywords(context)) > 0:
            dimensionless_values = DimensionlessParser.get_dimensionless_numbers(context)
            if len(dimensionless_values) == 0:
                pass
            elif len(dimensionless_values) == 1:
                action.ph = dimensionless_values[0]
            else:
                action.ph = dimensionless_values[0]
                print(
                    "Warning: More than one dimentionless value was found for the pH, only the first one was considered"
                )
        list_of_actions: List[Dict[str, Any]] = []
        if len(chemicals_info.chemical_list) == 0:
            pass
        elif len(chemicals_info.chemical_list[0].name.lower()) < 2:
            pass
        elif len(chemicals_info.chemical_list) == 1:
            if chemicals_info.chemical_list[0].name.lower() == "aqueous solution":
                chemicals_info.chemical_list[0].name = "water"
            action.material = chemicals_info.chemical_list[0]
            action.dropwise = chemicals_info.dropwise[0]
            list_of_actions.append(action.generate_dict())
        else:
            i = 0
            for chemical in chemicals_info.chemical_list:
                if chemical.name.lower() == "aqueous solution":
                    chemical.name = "water"
                action.material = chemical
                action.dropwise = chemicals_info.dropwise[i]
                list_of_actions.append(action.generate_dict())
            i += 1
        return list_of_actions


class CollectLayer(Actions):
    layer: Optional[str] = None

    @validator("layer")
    def layer_options(cls, layer):
        valid_layers = ["aqueous", "organic", None]
        if layer not in valid_layers:
            raise ValueError('layer must be equal to "aqueous" or "organic"')
        return layer

    @classmethod
    def generate_action(
        cls,
        context: str,
        parser_aqueous: KeywordSearching,
        parser_organic: KeywordSearching,
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="CollectLayer", action_context=context)
        aqueous_keywords = parser_aqueous.find_keywords(action.action_context)
        organic_keywords = parser_organic.find_keywords(action.action_context)
        if len(aqueous_keywords) > 0:
            action.layer = "aqueous"
        elif len(organic_keywords) > 0:
            action.layer = "organic"
        else:
            return []
        return [action.generate_dict()]


class Concentrate(Actions):
    @classmethod
    def generate_action(cls, context: str) -> List[Dict[str, Any]]:
        return [
            cls(
                action_name="Concentrate", action_context=context
            ).generate_dict()
        ]


class Degas(ActionsWithConditons):
    atmosphere: List[str] = []
    duration: Optional[str] = None

    @classmethod
    def generate_action(
        cls, context: str, conditions_parser: ParametersParser
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="Degas", action_context=context)
        action.validate_conditions(conditions_parser)
        return [action.generate_dict()]


class DrySolid(ActionsWithConditons):
    """Dry a solid under air or vacuum.
    For drying under vacuum, the atmosphere variable should contain the string 'vacuum'.
    For drying on air, the atmosphere variable should contain the string 'air'.
    For other atmospheres, the corresponding gas name should be given ('N2', 'argon', etc.).
    """

    duration: Optional[str] = None
    temperature: Optional[str] = None
    atmosphere: List[str] = []

    @classmethod
    def generate_action(
        cls, context: str, conditions_parser: ParametersParser
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="DrySolid", action_context=context)
        action.validate_conditions(conditions_parser)
        return [action.generate_dict()]


class DrySolution(ActionsWithChemicalAndConditions):
    """Dry an organic solution with a desiccant"""

    material: Optional[str] = None

    @classmethod
    def generate_action(
        cls,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        conditions_parser: ParametersParser,
        banned_parser: KeywordSearching
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="DrySolution", action_context=context)
        chemicals_info = action.validate_chemicals(
            schemas, schema_parser, amount_parser, action.action_context, banned_parser,
        )
        if len(chemicals_info.chemical_list) == 0:
            return DrySolid.generate_action(context, conditions_parser)
        elif len(chemicals_info.chemical_list) == 1:
            action.material = chemicals_info.chemical_list[0].name
        else:
            action.material = chemicals_info.chemical_list[0].name
            print(
                "Warning: More than one Material found on DrySolution object, only the first one was considered"
            )
        return [action.generate_dict()]


class Extract(ActionsWithchemicals):
    solvent: Optional[Chemical] = None
    repetitions: int = 1

    @classmethod
    def generate_action(
        cls,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        banned_parser: KeywordSearching
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="Extract", action_context=context)
        chemicals_info = action.validate_chemicals(
            schemas, schema_parser, amount_parser, action.action_context, banned_parser,
        )
        if len(chemicals_info.chemical_list) == 0:
            pass
        elif len(chemicals_info.chemical_list) == 1:
            action.solvent = chemicals_info.chemical_list[0]
            action.repetitions = chemicals_info.repetitions
        else:
            action.solvent = chemicals_info.chemical_list[0]
            action.repetitions = chemicals_info.repetitions
            print(
                "Warning: More than one Solvent found on DrySolution object, only the first one was considered"
            )
        return [action.generate_dict()]


class Filter(Actions):
    """
    Filtration action, possibly with information about what phase to keep ('filtrate' or 'precipitate')
    """

    phase_to_keep: Optional[str] = None

    @validator("phase_to_keep")
    def phase_options(cls, phase_to_keep):
        if phase_to_keep is not None and phase_to_keep not in [
            "filtrate",
            "precipitate",
            None,
        ]:
            raise ValueError(
                'phase_to_keep must be equal to "filtrate" or "precipitate"'
            )
        return phase_to_keep

    @classmethod
    def generate_action(
        cls,
        context: str,
        filtrate_parser: KeywordSearching,
        precipitate_parser: KeywordSearching,
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="Filter", action_context=context)
        filtrate_results = filtrate_parser.find_keywords(action.action_context)
        precipitate_results = precipitate_parser.find_keywords(action.action_context)
        if len(filtrate_results) > 0:
            action.phase_to_keep = "filtrate"
        elif len(precipitate_results) > 0:
            action.phase_to_keep = "precipitate"
        return [action.generate_dict()]
    
class Centrifuge(Actions):
    """
    Filtration action, possibly with information about what phase to keep ('filtrate' or 'precipitate')
    """

    phase_to_keep: Optional[str] = None

    @validator("phase_to_keep")
    def phase_options(cls, phase_to_keep):
        if phase_to_keep is not None and phase_to_keep not in [
            "filtrate",
            "precipitate",
            None,
        ]:
            raise ValueError(
                'phase_to_keep must be equal to "filtrate" or "precipitate"'
            )
        return phase_to_keep
    

    @classmethod
    def generate_action(
        cls,
        context: str,
        filtrate_parser: KeywordSearching,
        precipitate_parser: KeywordSearching,
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="Centrifuge", action_context=context)
        filtrate_results = filtrate_parser.find_keywords(action.action_context)
        precipitate_results = precipitate_parser.find_keywords(action.action_context)
        if len(filtrate_results) > 0:
            action.phase_to_keep = "filtrate"
        elif len(precipitate_results) > 0:
            action.phase_to_keep = "precipitate"
        return [action.generate_dict()]



class MakeSolution(ActionsWithChemicalAndConditions):
    """
    Action to make a solution out of a list of compounds.
    This action is usually followed by another action using it (Add, Quench, etc.).
    """

    materials: List[Chemical] = []
    dropwise: bool = False
    temperature: Optional[str] = None
    atmosphere: List[str] = []
    duration: Optional[str] = None
    pressure: Optional[str] = None

    @validator("materials")
    def amount_materials(cls, materials):
        if len(materials) < 2:
            raise ValueError(
                f"MakeSolution requires at least two components (actual: {len(materials)}"
            )
        return materials
    
    @classmethod
    def generate_action(
        cls,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        conditions_parser: ParametersParser,
        ph_parser: KeywordSearching,
        banned_parser: KeywordSearching,
        complex_parser=None
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="MakeSolution", action_context=context)
        action.validate_conditions(conditions_parser)
        chemicals_info = action.validate_chemicals(
            schemas, schema_parser, amount_parser, action.action_context, banned_parser)
        if len(chemicals_info.chemical_list) == 0:
            pass
        elif len(chemicals_info.chemical_list) == 1:
            return Add.generate_action(
                context,
                schemas,
                schema_parser,
                amount_parser,
                conditions_parser,
                ph_parser,
                banned_parser,
            )
        else:
            action.materials = chemicals_info.chemical_list
            for test in chemicals_info.dropwise:
                if test is True:
                    action.dropwise = True
                    break
        return [action.generate_dict()]
    

class Microwave(ActionsWithConditons):
    duration: Optional[str] = None
    temperature: Optional[str] = None

    @classmethod
    def generate_action(
        cls, context: str, conditions_parser: ParametersParser
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="Microwave", action_context=context)
        action.validate_conditions(conditions_parser)
        return [action.generate_dict()]


class Partition(ActionsWithchemicals):
    material_1: Optional[Chemical] = None
    material_2: Optional[Chemical] = None

    @classmethod
    def generate_action(
        cls,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        banned_parser: KeywordSearching
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="Partition", action_context=context)
        chemicals_info = action.validate_chemicals(
            schemas, schema_parser, amount_parser, action.action_context, banned_parser
        )
        if len(chemicals_info.chemical_list) == 0:
            pass
        elif len(chemicals_info.chemical_list) == 1:
            action.material_1 = chemicals_info.chemical_list[0]
        else:
            action.material_1 = chemicals_info.chemical_list[0]
            action.material_2 = chemicals_info.chemical_list[1]
        if len(chemicals_info.chemical_list) > 2:
            print(
                "Warning: More than two Materials have been found on Partition object, only the first two were considered"
            )
        return [action.generate_dict()]


class PhaseSeparation(Actions):
    @classmethod
    def generate_action(
        cls,
        context: str,
        filtrate_parser: KeywordSearching,
        precipitate_parser: KeywordSearching,
        centrifuge_parser: KeywordSearching,
        filter_parser: KeywordSearching,

    ) -> List[Dict[str, Any]]:
        action = cls(action_name="PhaseSeparation", action_context=context)
        filter_results = filter_parser.find_keywords(action.action_context)
        centrifuge_results = centrifuge_parser.find_keywords(action.action_context)
        if len(filter_results) > 0:
            return Filter.generate_action(context, filtrate_parser, precipitate_parser)
        elif len(centrifuge_results) > 0:
            return Centrifuge.generate_action(context, filtrate_parser, precipitate_parser)
        else:
            return [action.generate_dict()]


class Purify(Actions):
    @classmethod
    def generate_action(cls, context: str) -> List[Dict[str, Any]]:
        return [
            cls(action_name="Purify", action_context=context).generate_dict()
        ]


class Quench(ActionsWithChemicalAndConditions):
    material: Optional[Chemical] = None
    dropwise: bool = False
    temperature: Optional[str] = None

    @classmethod
    def generate_action(
        cls,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        conditions_parser: ParametersParser,
        ph_parser: KeywordSearching,
        banned_parser: KeywordSearching
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="Quench", action_context=context)
        action.validate_conditions(conditions_parser)
        chemicals_info = action.validate_chemicals(
            schemas, schema_parser, amount_parser, action.action_context, banned_parser
        )
        if len(chemicals_info.chemical_list) == 0:
            pass
        elif len(chemicals_info.chemical_list) == 1:
            action.material = chemicals_info.chemical_list[0]
            action.dropwise = chemicals_info.dropwise[0]
        else:
            action.material = chemicals_info.chemical_list[0]
            action.dropwise = chemicals_info.dropwise[0]
            print(
                "Warning: More than one Material found on Quench object, only the first one was considered"
            )
        if len(ph_parser.find_keywords(context)) > 0:
            new_action: List[Dict[str, Any]] = PH.generate_action(
                context, schemas, schema_parser, amount_parser, conditions_parser
            )
            return [action.generate_dict()] + new_action
        return [action.generate_dict()]


class Recrystallize(ActionsWithChemicalAndConditions):
    solvent: Optional[Chemical] = None
    temperature: Optional[str] = None
    atmosphere: List[str] = []
    duration: Optional[str] = None
    pressure: Optional[str] = None

    @classmethod
    def generate_action(
        cls,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        conditions_parser: ParametersParser,
        banned_parser: KeywordSearching
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="Recrystallize", action_context=context)
        action.validate_conditions(conditions_parser)
        chemicals_info = action.validate_chemicals(
            schemas, schema_parser, amount_parser, action.action_context, banned_parser
        )
        if len(chemicals_info.chemical_list) == 0:
            pass
        elif len(chemicals_info.chemical_list) == 1:
            action.solvent = chemicals_info.chemical_list[0]
        else:
            action.solvent = chemicals_info.chemical_list[0]
            print(
                "Warning: More than one Solvent found on Recrystallize object, only the first one was considered"
            )
        return [action.generate_dict()]


class Reflux(ActionsWithConditons):
    duration: Optional[str] = None
    dean_stark: bool = False
    atmosphere: List[str] = []

    @classmethod
    def generate_action(
        cls, context: str, conditions_parser: ParametersParser
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="Reflux", action_context=context)
        action.validate_conditions(conditions_parser)
        return [action.generate_dict()]

    
class Stir(ActionsWithConditons):
    duration: Optional[str] = None
    stirring_speed: Optional[str] = None
    temperature: Optional[str] = None
    atmosphere: List[str] = []
    pressure: Optional[str] = None
    
    @classmethod
    def generate_action(
        cls, context: str, conditions_parser: ParametersParser, complex_conditions_parser: ComplexParametersParser
    ) -> List[Dict[str, Any]]:
        action: Stir = cls(action_name="Stir", action_context=context)
        action.validate_conditions(conditions_parser, complex_conditions_parser=complex_conditions_parser)
        list_of_actions: List[Any] = []
        if action.duration is not None:
            list_of_actions.append(action.generate_dict())
        return list_of_actions


class SetTemperature(ActionsWithConditons):
    """
    If there is a duration given with cooling/heating, use "Stir" instead
    """

    temperature: Optional[str] = None
    microwave: bool = False
    heat_ramp: Optional[str] = None
    duration: Optional[str] = None
    pressure: Optional[str] = None
    atmosphere: List[str] = []
    stirring_speed: Optional[str] = None

    @classmethod
    def generate_action(
        cls,
        context: str,
        conditions_parser: ParametersParser,
        complex_conditions_parser: ComplexParametersParser,
        microwave_parser: KeywordSearching,
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="SetTemperature", action_context=context)
        action.validate_conditions(conditions_parser, complex_conditions_parser=complex_conditions_parser)
        keywords_list = microwave_parser.find_keywords(context)
        if len(keywords_list) > 0:
            action.microwave = True
        return [action.generate_dict()]

class ReduceTemperature(ActionsWithConditons):
    """
    If there is a duration given with cooling/heating, use "Stir" instead
    """

    temperature: Optional[str] = None
    microwave: bool = False
    heat_ramp: Optional[str] = None
    duration: Optional[str] = None
    pressure: Optional[str] = None
    atmosphere: List[str] = []
    stirring_speed: Optional[str] = None

    @classmethod
    def generate_action(
        cls,
        context: str,
        conditions_parser: ParametersParser,
        complex_conditions_parser: ComplexParametersParser,
        microwave_parser: KeywordSearching,
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="SetTemperature", action_context=context)
        action.validate_conditions(conditions_parser, complex_conditions_parser=complex_conditions_parser)
        if action.temperature is None:
            action.temperature == "cool"
        keywords_list = microwave_parser.find_keywords(context)
        if len(keywords_list) > 0:
            action.microwave = True
        return [action.generate_dict()]

class Sonicate(ActionsWithConditons):
    duration: Optional[str] = None
    temperature: Optional[str] = None

    @classmethod
    def generate_action(
        cls, context: str, conditions_parser: ParametersParser
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="Sonicate", action_context=context)
        action.validate_conditions(conditions_parser)
        return [action.generate_dict()]


class Triturate(ActionsWithchemicals):
    solvent: Optional[Chemical] = None

    @classmethod
    def generate_action(
        cls,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        banned_parser
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="Triturate", action_context=context)
        chemicals_info = action.validate_chemicals(
            schemas, schema_parser, amount_parser, action.action_context, banned_parser
        )
        if len(chemicals_info.chemical_list) == 0:
            pass
        elif len(chemicals_info.chemical_list) == 1:
            action.solvent = chemicals_info.chemical_list[0]
        else:
            action.solvent = chemicals_info.chemical_list[0]
            print(
                "Warning: More than one Solvent found on Triturate object, only the first one was considered"
            )
        return [action.generate_dict()]


class Wait(ActionsWithConditons):
    """
    NB: "Wait" as an action can be ambiguous depending on the context.
    It seldom means "waiting without doing anything", but is often "continue what was before", at least in Pistachio.
    """

    duration: Optional[str] = None
    temperature: Optional[str] = None

    @classmethod
    def generate_action(
        cls, context: str, conditions_parser: ParametersParser
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="Wait", action_context=context)
        action.validate_conditions(conditions_parser)
        action_list: List[Dict[str, Any]] = []
        if action.duration is not None:
            action_list.append(action.generate_dict())
        return action_list


class Wash(ActionsWithChemicalAndConditions):
    material: Optional[Chemical] = None
    temperature: Optional[str] = None
    duration: Optional[str] = None
    method: Optional[str] = None
    repetitions: int = 1

    @classmethod
    def generate_action(
        cls,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        conditions_parser: ParametersParser,
        centrifuge_parser: KeywordSearching,
        filter_parser: KeywordSearching,
        banned_parser: KeywordSearching,
        complex_parser: ComplexParametersParser=None
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="Wash", action_context=context)
        action.validate_conditions(conditions_parser)
        chemicals_info: ChemicalInfo = action.validate_chemicals(
            schemas, schema_parser, amount_parser, 
            action.action_context, banned_parser, complex_parser=complex_parser
        )
        centrifuge_results: List[str] = centrifuge_parser.find_keywords(action.action_context)
        filter_results: List[str] = filter_parser.find_keywords(action.action_context)
        list_of_actions: List[Any] = []
        if len(filter_results) > 0:
            action.method = "filtration"
        elif len(centrifuge_results) > 0:
            action.method = "centrifugation"
        if len(chemicals_info.chemical_list) == 0:
            pass
        elif len(schemas) == 1:
            action.material = chemicals_info.chemical_list[0]
            action.repetitions = chemicals_info.repetitions
            list_of_actions.append(action.generate_dict())
        else:
            for material in chemicals_info.chemical_list:
                action.material = material
                action.repetitions = chemicals_info.repetitions
                list_of_actions.append(action.generate_dict())
        number_list: List[str] = DimensionlessParser.get_dimensionless_numbers(context)
        if action.repetitions == 1:
            if len(number_list) == 0:
                pass
            elif len(number_list) == 1:
                action.amount = int(float(number_list[0]))
            else:
                action.amount = int(float(number_list[0]))
                print(
                    "Warning: More than one adimensional number was found, only the first one was considered"
                    )
            list_of_actions: List[Any] = []
            if 6 > action.amount > 1:
                list_of_actions.append(action.generate_dict())
        return list_of_actions

class Yield(ActionsWithchemicals):
    material: Optional[Chemical] = None

    @classmethod
    def generate_action(
        cls,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        banned_parser: KeywordSearching
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="Yield", action_context=context)
        chemicals_info = action.validate_chemicals(
            schemas, schema_parser, amount_parser, action.action_context, banned_parser
        )
        if len(chemicals_info.chemical_list) == 0:
            pass
        elif len(chemicals_info.chemical_list) == 1:
            action.material = chemicals_info.chemical_list[0]
        else:
            action.material = chemicals_info.chemical_list[0]
            print(
                "Warning: More than one Material found on Yield object, only the first one was considered"
            )
        return [action.generate_dict()]


class NewSolution(ActionsWithChemicalAndConditions):
    solution: Optional[Chemical] = None

    @classmethod
    def generate_action(
        cls,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        conditions_parser: ParametersParser,
        ph_parser: KeywordSearching,
        banned_parser: KeywordSearching,
        complex_parser: ComplexParametersParser=None,
    ) -> List[Dict[str, Any]]:
        action: NewSolution = cls(action_name="NewSolution", action_context=context)
        chemicals_info: ChemicalInfo = action.validate_chemicals(
            schemas, schema_parser, amount_parser, action.action_context, banned_parser, complex_parser=complex_parser
        )
        if chemicals_info.final_solution is not None:
            action.solution = chemicals_info.final_solution
        list_of_actions: List[Dict[str, Any]] = []
        list_of_actions.append(action.generate_dict())
        add_actions = Add.generate_action(
                context, schemas, schema_parser, amount_parser, conditions_parser, ph_parser, banned_parser
            )
        list_of_actions += add_actions
        return list_of_actions

class Crystallization(ActionsWithConditons):
    temperature: Optional[str] = None
    duration: Optional[str] = None
    pressure: Optional[str] = None
    stirring_speed: Optional[str] = None
    microwave: bool = False

    @classmethod
    def generate_action(
        cls, context: str, conditions_parser: ParametersParser, complex_conditions_parser: ComplexParametersParser, microwave_parser: KeywordSearching
    ) -> List[Dict[str, Any]]:
        action: Crystallization = cls(action_name="Crystallization", action_context=context)
        action.validate_conditions(conditions_parser, complex_conditions_parser=complex_conditions_parser)
        keywords_list = microwave_parser.find_keywords(context)
        if len(keywords_list) > 0:
            action.microwave = True
        return [action.generate_dict()]

class Separate(ActionsWithConditons):
    temperature: Optional[str] = None
    phase_to_keep: str = "precipitate"
    method: Optional[str] = None

    @classmethod
    def generate_action(
        cls,
        context: str,
        conditions_parser: ParametersParser,
        filtrate_parser: KeywordSearching,
        precipitate_parser: KeywordSearching,
        centrifuge_parser: KeywordSearching,
        filter_parser: KeywordSearching,
        evaporation_parser: KeywordSearching
    ) -> List[Dict[str, Any]]:
        action: Separate = cls(action_name="Separate", action_context=context)
        action.validate_conditions(conditions_parser)
        filtrate_results: List[str] = filtrate_parser.find_keywords(action.action_context)
        precipitate_results: List[str] = precipitate_parser.find_keywords(action.action_context)
        centrifuge_results: List[str] = centrifuge_parser.find_keywords(action.action_context)
        filter_results: List[str] = filter_parser.find_keywords(action.action_context)
        evaporation_results: List[str] = evaporation_parser.find_keywords(action.action_context)
        if len(filtrate_results) > 0:
            action.phase_to_keep = "filtrate"
        elif len(precipitate_results) > 0:
            action.phase_to_keep = "precipitate"
        if len(filter_results) > 0:
            action.method = "filtration"
        elif len(centrifuge_results) > 0:
            action.method = "centrifugation"
        elif len(evaporation_results) > 0:
            action = Dry(action_name = "Dry", temperature=action.temperature)
        return [action.generate_dict()]
        



class Dry(ActionsWithConditons):
    temperature: Optional[str] = None
    duration: Optional[str] = None
    atmosphere: List[str] = []

    @classmethod
    def generate_action(
        cls, context: str, conditions_parser: ParametersParser
    ) -> List[Dict[str, Any]]:
        action: Dry = cls(action_name="Dry", action_context=context)
        action.validate_conditions(conditions_parser)
        return [action.generate_dict()]

class ThermalTreatment(ActionsWithConditons):
    temperature: Optional[str] = None
    duration: Optional[str] = None
    heat_ramp: Optional[str] = None
    atmosphere: List[str] = []
    flow_rate: Optional[str] = None
    @classmethod
    def generate_action(
        cls, context: str, conditions_parser: ParametersParser, complex_conditions_parser: ComplexParametersParser
    ) -> List[Dict[str, Any]]:
        action: ThermalTreatment = cls(action_name="ThermalTreatment", action_context=context)
        action.validate_conditions(conditions_parser, complex_conditions_parser=complex_conditions_parser)
        return [action.generate_dict()]

class IonExchange(Treatment):
    
    @classmethod
    def generate_action(
        cls,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        conditions_parser: ParametersParser,
        banned_parser: KeywordSearching
    ) -> List[Dict[str, Any]]:
        return Treatment.generate_treatment("IonExchange", context, schemas, schema_parser, amount_parser, conditions_parser, banned_parser)
    
class AlkalineTreatment(Treatment):
    
    @classmethod
    def generate_action(
        cls,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        conditions_parser: ParametersParser,
        banned_parser: KeywordSearching
    ) -> List[Dict[str, Any]]:
        return Treatment.generate_treatment("AlkalineTreatment", context, schemas, schema_parser, amount_parser, conditions_parser, banned_parser)
    
class AcidTreatment(Treatment):

    @classmethod
    def generate_action(
        cls,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        conditions_parser: ParametersParser,
        banned_parser: KeywordSearching
    ) -> List[Dict[str, Any]]:
        return Treatment.generate_treatment("AcidTreatment", context, schemas, schema_parser, amount_parser, conditions_parser, banned_parser)

class Repeat(Actions):
    amount: str = 1
    
    @classmethod
    def generate_action(cls, context: str):
        action: Repeat = cls(action_name="Repeat", action_context=context)
        number_list: List[str] = DimensionlessParser.get_dimensionless_numbers(context)
        if len(number_list) == 0:
            pass
        elif len(number_list) == 1:
            action.amount = int(float(number_list[0]))
        else:
            action.amount = int(float(number_list[0]))
            print(
                "Warning: More than one adimensional number was found, only the first one was considered"
                )
        list_of_actions: List[Any] = []
        if 6 > action.amount > 1:
            list_of_actions.append(action.generate_dict())
        return list_of_actions
    
class Transfer(Actions):
    recipient: str = ""
    
    @classmethod
    def generate_action(cls, context: str, schemas: List[str], schemas_parser: SchemaParser, banned_transfer_parser: KeywordSearching):
        action: Transfer = cls(action_name="Transfer", action_context=context)
        if len(schemas) == 0:
            pass
        elif len(schemas) == 1:
            name: List[str] = schemas_parser.get_atribute_value(schemas[0], "type")
            banned_keywords_name: List[str] = banned_transfer_parser.find_keywords(name[0].lower())
            if len(banned_keywords_name) > 0:
                final_name = ""
            else:
                final_name = name[0]
            size: List[str]= schemas_parser.get_atribute_value(schemas[0], "volume")
            banned_keywords_size: List[str] = banned_transfer_parser.find_keywords(size[0].lower())
            if len(banned_keywords_size) > 0:
                final_size = size[0]
            else:
                final_size = size[0] + " "
            action.recipient = f"{final_size}{final_name}".strip()
        else:
            name: str = schemas_parser.get_atribute_value(schemas[0], "type")
            banned_keywords_name = banned_transfer_parser.find_keywords(name[0].lower())
            if len(banned_keywords_name) > 0:
                final_name = ""
            else:
                final_name = name[0]
            size = schemas_parser.get_atribute_value(schemas[0], "volume")
            banned_keywords_size: List[str] = banned_transfer_parser.find_keywords(size[0].lower())
            if len(banned_keywords_size) > 0:
                final_size = size[0]
            else:
                final_size = size[0] + " "
            action.recipient = f"{final_size}{final_name}".strip()
            print(
                "Warning: More than one recipient was found, only the first one was considered"
                )
        return [action.generate_dict()]


class SetAtmosphere(Actions):
    atmosphere: List[str] = []
    pressure: Optional[str] = None
    flow_rate: Optional[str] = None

class MicrowaveMaterial(ActionsWithConditons):
    pass

class Grind(ActionsWithConditons):
    size: Optional[str] = None

    @classmethod
    def generate_action(cls, context: str, conditions_parser: ParametersParser):
        action: Grind = cls(action_name="Grind", action_context=context)
        action.validate_conditions(conditions_parser, add_others=True)
        list_of_actions: List[Dict[str, Any]] = [action.generate_dict()]
        if action.size is not None:
            list_of_actions.append(Sieve(action_name="Sieve", size=action.size).generate_dict())
        return list_of_actions

class Sieve(ActionsWithConditons):
    size: Optional[str] = None
    @classmethod
    def generate_action(cls, context: str, conditions_parser: ParametersParser):
        action: Sieve = cls(action_name="Sieve", action_context=context)
        action.validate_conditions(conditions_parser, add_others=True)
        return [action.generate_dict()]

BANNED_TRANSFER_REGISTRY: List[str] = ["N/A"]

BANNED_CHEMICALS_REGISTRY: List[str] = [
    "reaction",
    "title",
    "newsolution",
    "extract",
    "heated",
    "cooled",
    "hydrothermal",
    "unknown",
    "rinse",
    "teflon",
    "Te\ue104on",
    "teflon-lined",
    "autoclave",
    "washing",
    "wash",
    "mixed",
    "precursor",
    "pre-prepared",
    "prepapared",
    "clear",
    "obtained",
    "new",
    "reactants",
    "reactant",
    "N/A",
    "N/A,\n",
    "ph",
    "final",
    "suspension",
    "makesolution",
    "rising",
    "slurry",
    "leaching",
    "metal",
    "name",
    "Polytetrafluoroethylene",
    "PTFE",
    "mixture",
    "derivative",
    "layer",
    "layers",
    "step",
    "residue",
    "phase",
    "prepared",
    "neutralized",
    "basified"
]

ACTION_REGISTRY: Dict[str, Any] = {
    "add": Add,
    "cool": SetTemperature,
    "heat": SetTemperature,
    "settemperature": SetTemperature,
    "stir": Stir,
    "concentrate": Concentrate,
    "evaporate": Concentrate,
    "drysolution": DrySolution,
    "dry": DrySolution,
    "collectlayer": CollectLayer,
    "collect": CollectLayer,
    "extract": Extract,
    "wash": Wash,
    "makesolution": MakeSolution,
    "filter": Filter,
    "recrystallize": Recrystallize,
    "crystallize": Recrystallize,
    "recrystalize": Recrystallize,
    "purify": Purify,
    "quench": Quench,
    "phaseseparation": PhaseSeparation,
    "adjustph": PH,
    "reflux": Reflux,
    "drysolid": DrySolid,
    "degas": Degas,
    "partition": Partition,
    "sonicate": Sonicate,
    "triturate": Triturate,
    "wait": Wait,
    "finalproduct": Yield,
}
PISTACHIO_ACTION_REGISTRY: Dict[str, Any] = {
    "add": Add,
    "cool": ReduceTemperature,
    "heat": SetTemperature,
    "settemperature": SetTemperature,
    "stir": Stir,
    "concentrate": Concentrate,
    "drysolution": DrySolution,
    "dry": DrySolution,
    "extract": Extract,
    "wash": Wash,
    "makesolution": Add,
    "purify": Purify,
    "quench": Quench,
    "phaseseparation": PhaseSeparation,
    "partition": Partition,
    "wait": Wait,
    "ph": Add,
    "sonicate": Sonicate,
    "degas": Degas,
    "recrystallize": Recrystallize,
    "triturate": Triturate
}
ORGANIC_ACTION_REGISTRY: Dict[str, Any] = {
    "add": Add,
    "pour": Add,
    "dissolve": Add,
    "dilute": Add,
    "cool": SetTemperature,
    "heat": SetTemperature,
    "settemperature": SetTemperature,
    "warm": SetTemperature,
    "stir": Stir,
    "concentrate": Concentrate,
    "evaporate": Concentrate,
    "remove": Concentrate,
    "drysolution": DrySolution,
    "dry": DrySolution,
    "solidify": DrySolution,
    "collectlayer": CollectLayer,
    "collect": CollectLayer,
    "extract": Extract,
    "wash": Wash,
    "filter": Filter,
    "recrystallize": Recrystallize,
    "crystallize": Recrystallize,
    "recrystalize": Recrystallize,
    "purify": Purify,
    "distill": Purify,
    "quench": Quench,
    "phaseseparation": PhaseSeparation,
    "adjustph": PH,
    "reflux": Reflux,
    "drysolid": DrySolid,
    "degas": Degas,
    "partition": Partition,
    "sonicate": Sonicate,
    "wait": Wait,
    "finalproduct": Yield,
    "finalproduct:": Yield,
    "provide": Yield,
    "afford": Yield,
    "obtain": Yield,
}
MATERIAL_ACTION_REGISTRY: Dict[str, Any] = {
    "add": Add,
    "newsolution": NewSolution,
    "makesolution": NewSolution,
    "newmixture": NewSolution,
    "crystallization": Crystallization,
    "separate": Separate,
    "sonicate": Stir,
    "wash": Wash,
    "wait": Wait,
    "dry": Dry,
    "calcination": ThermalTreatment,
    "stir": Stir,
    "ionexchange": IonExchange,
    "ion-exchange": IonExchange,
    "ion exchange": IonExchange,
    "alkalinetreatment": AlkalineTreatment,
    "acidtreatment": AcidTreatment,
    "repeat": Repeat,
    "cool": ReduceTemperature,
    "heat": SetTemperature,
    "settemperature":  SetTemperature,
    "grind": Grind,
    "sieve": Sieve,
    "extract": Wash,
    "quench": Wash,
    "thermaltreatment": ThermalTreatment,
    "posttreatment": ThermalTreatment, 
    "drysolid": Dry,
    "drysolution": Dry,
    "dry": Dry,
    "concentrate": Separate,
    "centrifugate": Separate,
    "filter": Separate,
    "sonicate": Sonicate,
    "reflux": SetTemperature,
    "phaseseparation": Separate,
    "purify": Wash,
    "transfer": None,
    "degas": None,
    "invalidaction": None,
    "recrystallize": None,
    "followotherprocedure": None,
    "synthesisproduct": None,
    "synthesismethod": None,
    "synthesisvariant": None,
    "yield": None,
    "noaction": None,

}

ELEMENTARY_ACTION_REGISTRY: Dict[str, Any] = {
    "add": Add,
    "newsolution": NewSolution,
    "makesolution": NewSolution,
    "newmixture": NewSolution,
    "separate": Separate,
    "wash": Wash,
    "wait": Wait,
    "stir": Stir,
    "repeat": Repeat,
    "cool": ReduceTemperature,
    "heat": SetTemperature,
    "grind": Grind,
    "sieve": Sieve,
}

SAC_ACTION_REGISTRY: Dict[str, Any] = {
    "add": Add,
    "makesolution": MakeSolution,
    "newsolution": MakeSolution,
    "separate": PhaseSeparation,
    "centrifugate": PhaseSeparation,
    "filter": PhaseSeparation,
    "concentrate": Dry,
    "cool": ReduceTemperature,
    "heat": SetTemperature,
    "wash": Wash,
    "wait": Wait,
    "reflux": SetTemperature,
    "drysolid": Dry,
    "drysolution": Dry,
    "dry": Dry,
    "posttreatment": ThermalTreatment,
    "thermaltreatment": ThermalTreatment,
    "stir": Stir,
    "sonicate": Sonicate,
    "quench": Quench,
    "settemperature": SetTemperature,
    "grind": Grind,
    "sieve": Sieve,
    "anneal": ThermalTreatment,
    "calcine": ThermalTreatment,
    "phaseseparation": PhaseSeparation,
    "degas": Degas,
    "extract": Extract,
    "purify": Purify,
    "synthesisproduct": None,
    "synthesismethod": None,
    "synthesisvariant": None,
    "yield": None,
    "noaction": None,
    "transfer": Transfer,
    "invalidaction": None,
    "recrystallize": None,
    "followotherprocedure": None

}
AQUEOUS_REGISTRY: List[str] = ["aqueous", "aq", "hydrophilic", "water", "aquatic"]
ORGANIC_REGISTRY: List[str] = ["organic", "org", "hydrophobic"]
FILTRATE_REGISTRY: List[str] = [
    "filtrate",
    "lixiviate",
    "percolate",
    "permeate",
    "liquid",
]
PRECIPITATE_REGISTRY: List[str] = [
    "precipitate",
    "residue",
    "filter cake",
    "sludge",
    "solid",
    "powder"
]
FILTER_REGISTRY: List[str] = [
    "filtrate",
    "filter",
    "filtration"
]
CENTRIFUGATION_REGISTRY: List[str] = [
    "centrifuge",
    "centrifugation",
    "centrifugally",
    "centrifugal",
    "centrifugate"
]
EVAPORATION_REGISTRY: List[str] = [
    "evaporation",
    "evaporate",
    "evaporator",
    "concentrate"
]
MICROWAVE_REGISTRY: List[str] = ["microwave", "microwaves"]
PH_REGISTRY: List[str] = ["ph"]