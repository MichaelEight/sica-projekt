from dataclasses import dataclass, field
from datetime import datetime
import uuid
import json


def auto_label(type: str, value_ms: float | None = None, t1: float = 0.0, t2: float = 0.0, category: str = "", probs: dict | None = None, label: str = "") -> str:
    """Generate a display label based on marking type and values."""
    try:
        if type == "pr" and value_ms is not None:
            return f"PR: {value_ms:.0f} ms"
        elif type == "qrs" and value_ms is not None:
            return f"QRS: {value_ms:.0f} ms"
        elif type == "qt" and value_ms is not None:
            return f"QT: {value_ms:.0f} ms"
        elif type == "rr" and value_ms is not None:
            hr = 60000.0 / value_ms if value_ms > 0 else 0.0
            return f"R-R: {value_ms:.0f} ms ({hr:.0f} bpm)"
        elif type == "annotation":
            return category if category else "Annotation"
        elif type == "scan" and probs:
            top_class = max(probs, key=probs.get)
            pct = probs[top_class] * 100
            # Use Polish class names if available
            _CLASS_NAMES_PL = {
                "class_healthy": "Zdrowy (NORM)",
                "class_front_heart_attack": "Zawał przedniej ściany",
                "class_side_heart_attack": "Zawał ściany bocznej",
                "class_bottom_heart_attack": "Zawał ściany dolnej",
                "class_back_heart_attack": "Zawał ściany tylnej",
                "class_complete_right_conduction_disorder": "CRBBB",
                "class_incomplete_right_conduction_disorder": "IRBBB",
                "class_complete_left_conduction_disorder": "CLBBB",
            }
            name = _CLASS_NAMES_PL.get(top_class, top_class)
            return f"{name}: {pct:.0f}%"
        elif type == "custom":
            return label if label else "Custom"
        else:
            return label if label else type
    except Exception:
        return label if label else type


MARKING_STYLES = {
    "annotation": {"color": (139, 92, 246), "border": "dashed", "label_prefix": ""},
    "pr":         {"color": (249, 115, 22), "border": "solid",  "label_prefix": "PR"},
    "qrs":        {"color": (239, 68, 68),  "border": "solid",  "label_prefix": "QRS"},
    "qt":         {"color": (34, 197, 94),  "border": "solid",  "label_prefix": "QT"},
    "rr":         {"color": (59, 130, 246), "border": "solid",  "label_prefix": "R-R"},
    "custom":     {"color": (156, 163, 175), "border": "solid",  "label_prefix": ""},
    "scan":       {"color": None,           "border": "none",   "label_prefix": "AI"},
}


@dataclass
class Marking:
    type: str
    lead: str
    t1: float
    t2: float
    label: str = ""
    category: str = ""
    note: str = ""
    value_ms: float | None = None
    probs: dict | None = None
    color_code: int = 0
    source: str = "user"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        if not self.label:
            self.label = auto_label(
                self.type,
                value_ms=self.value_ms,
                t1=self.t1,
                t2=self.t2,
                category=self.category,
                probs=self.probs,
                label=self.label,
            )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "lead": self.lead,
            "t1": self.t1,
            "t2": self.t2,
            "label": self.label,
            "category": self.category,
            "note": self.note,
            "value_ms": self.value_ms,
            "probs": self.probs,
            "color_code": self.color_code,
            "source": self.source,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Marking":
        try:
            m = cls(
                type=d.get("type", "custom"),
                lead=d.get("lead", ""),
                t1=float(d.get("t1", 0.0)),
                t2=float(d.get("t2", 0.0)),
                label=d.get("label", ""),
                category=d.get("category", ""),
                note=d.get("note", ""),
                value_ms=d.get("value_ms"),
                probs=d.get("probs"),
                color_code=int(d.get("color_code", 0)),
                source=d.get("source", "user"),
            )
            if "id" in d:
                m.id = d["id"]
            if "created_at" in d:
                m.created_at = d["created_at"]
            return m
        except Exception:
            return cls(type="custom", lead="", t1=0.0, t2=0.0, label="(invalid)")


_MAX_UNDO = 10


class MarkingStore:
    def __init__(self):
        self._markings: list[Marking] = []
        self._undo_stack: list[tuple] = []
        self._redo_stack: list[tuple] = []

    # -- core operations --

    def add(self, marking: Marking) -> None:
        self._markings.append(marking)
        self._push_undo(("add", marking.to_dict(), None))

    def edit(self, marking_id: str, **kwargs) -> bool:
        m = self.get_by_id(marking_id)
        if m is None:
            return False
        old = m.to_dict()
        for k, v in kwargs.items():
            if hasattr(m, k) and k not in ("id", "created_at"):
                setattr(m, k, v)
        new = m.to_dict()
        self._push_undo(("edit", old, new))
        return True

    def delete(self, marking_id: str) -> bool:
        m = self.get_by_id(marking_id)
        if m is None:
            return False
        old = m.to_dict()
        self._markings = [x for x in self._markings if x.id != marking_id]
        self._push_undo(("delete", old, None))
        return True

    # -- undo / redo --

    def _push_undo(self, entry: tuple) -> None:
        self._undo_stack.append(entry)
        if len(self._undo_stack) > _MAX_UNDO:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        action, before, after = self._undo_stack.pop()
        try:
            if action == "add":
                # before contains the marking that was added
                self._markings = [x for x in self._markings if x.id != before["id"]]
            elif action == "delete":
                # before contains the marking that was deleted — restore it
                self._markings.append(Marking.from_dict(before))
            elif action == "edit":
                # before contains old state, restore it
                m = self.get_by_id(before["id"])
                if m is not None:
                    for k, v in before.items():
                        if hasattr(m, k):
                            setattr(m, k, v)
            self._redo_stack.append((action, before, after))
            return True
        except Exception:
            return False

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        action, before, after = self._redo_stack.pop()
        try:
            if action == "add":
                # re-add the marking
                self._markings.append(Marking.from_dict(before))
            elif action == "delete":
                # re-delete
                self._markings = [x for x in self._markings if x.id != before["id"]]
            elif action == "edit":
                # after contains the edited state — reapply
                m = self.get_by_id(after["id"])
                if m is not None:
                    for k, v in after.items():
                        if hasattr(m, k):
                            setattr(m, k, v)
            self._undo_stack.append((action, before, after))
            if len(self._undo_stack) > _MAX_UNDO:
                self._undo_stack.pop(0)
            return True
        except Exception:
            return False

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    # -- queries --

    def get_all(self) -> list[Marking]:
        return list(self._markings)

    def get_by_type(self, types: list[str]) -> list[Marking]:
        return [m for m in self._markings if m.type in types]

    def search(self, query: str) -> list[Marking]:
        if not query:
            return list(self._markings)
        q = query.lower()
        results = []
        for m in self._markings:
            if (
                q in m.label.lower()
                or q in m.note.lower()
                or q in m.lead.lower()
                or q in m.category.lower()
            ):
                results.append(m)
        return results

    def get_by_id(self, marking_id: str) -> Marking | None:
        for m in self._markings:
            if m.id == marking_id:
                return m
        return None

    # -- file I/O --

    def save_ann(self, path: str, patient: dict | None = None) -> bool:
        try:
            data = {
                "format": "ekg-assistant-ann",
                "version": 2,
                "markings": [m.to_dict() for m in self._markings],
            }
            if patient:
                data["patient"] = patient
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False

    def load_ann(self, path: str) -> tuple[bool, dict | None]:
        """Load .ann file. Returns (success, patient_dict_or_None)."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("version") != 2:
                return False, None
            self._markings = [Marking.from_dict(d) for d in data.get("markings", [])]
            self._undo_stack.clear()
            self._redo_stack.clear()
            patient = data.get("patient", None)
            return True, patient
        except Exception:
            return False, None

    def clear(self) -> None:
        self._markings.clear()
        self._undo_stack.clear()
        self._redo_stack.clear()
