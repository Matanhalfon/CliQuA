from typing import List
import dataclasses
import copy

@dataclasses.dataclass
class ModelAnnotation:
    # Raw model answer.
    answer: str 
    # Extracted match label. 
    label: bool
    # Attribution refenece to passage.
    reference: str
    # Answer explanation.
    explanation: str

    def __init__(self,
                 answer:str = "",
                 label:bool = False,
                 reference:str = "",
                 explanation:str = ""):
        self.answer=answer
        self.label=label
        self.reference=reference
        self.explanation=explanation


    @classmethod
    def from_dict(cls, data):
        if 'label' not in data:
            return ModelAnnotation("",False, "","")
        return cls(**data)

@dataclasses.dataclass
class AnnotatedPassage:
    # Passage index in original record.
    index: int
    # Passage content.
    passage: str
    # Ground truth attributions to passage.
    ground_truth_attributions: List[str]
    # Model annotations.
    model_annotation: ModelAnnotation
    
    def __init__(self,
                 index: int,
                 passage: str,
                 ground_truth_attributions: list[str] = [],
                 model_annotation:ModelAnnotation =None):
        self.index = index
        self.passage = passage
        self.ground_truth_attributions=copy.deepcopy(ground_truth_attributions)
        self.model_annotation = ModelAnnotation()
        if model_annotation:
            self.model_annotation=copy.deepcopy(model_annotation)


    @classmethod
    def from_dict(cls, data):
        model_annotation_data = data.pop('model_annotation', {})
        model_annotation = ModelAnnotation.from_dict(model_annotation_data)
        return cls(model_annotation=model_annotation, **data)

    def to_dict(self):
        return{
            'index': self.index,
            'passage': self.passage,
            'ground_truth_attributions': self.ground_truth_attributions,
            'model_annotation': self.model_annotation.__dict__,
        }


@dataclasses.dataclass
class AnnotatedSample:
    # Unique sample id.
    sample_id: str
    # File name.
    file_name: int
    # Evaluated query.
    query: str
    # Ground truth match indication.
    ground_truth_label: bool
    # Passage with model and ground truth annotations.
    annotated_passages: List[AnnotatedPassage]

    def to_dict(self):
        return{
            'sample_id': self.sample_id,
            'file_name': self.file_name,
            'query': self.query,
            'ground_truth_label': self.ground_truth_label,      
            'annotated_passages':  [
                item.to_dict() for item in self.annotated_passages]
        }

    @classmethod
    def from_dict(cls, data):
        if 'file_name' not in data:
            data['file_name']=int(data['sample_id'].split("_")[0])
        annotated_passages_data = data.pop('annotated_passages', [])
        annotated_passages = [
            AnnotatedPassage.from_dict(item) for item in annotated_passages_data
        ]
        return cls(annotated_passages=annotated_passages, **data)    