from typing import List, Dict, Any, Tuple


class DecisionEngine:
    """
    Deterministic rule engine that maps freshness classification (Grade A/B/C)
    and detected surface defects into operational sorting decisions and GPIO signals.
    """

    @staticmethod
    def evaluate(
        grade: str,
        grade_confidence: float,
        defects: List[Dict[str, Any]],
        confidence_threshold: float = 0.75
    ) -> Tuple[str, str, str]:
        """
        Evaluates fish inspection data against quality standards.

        Priority Rules:
        1. Grade C -> Immediate FAIL / RED (severe biological spoilage).
        2. Any defects present -> FAIL / RED (physical damage, lesions, parasites, abnormal color).
        3. Grade B and confidence < threshold -> CONDITIONAL / YELLOW (requires operator review).
        4. Grade A or Grade B (confident) -> PASS / GREEN (conveyor continues uninterrupted).

        Returns:
            Tuple of (decision, hardware_signal, reason_summary)
            decision: "PASS" | "CONDITIONAL" | "FAIL"
            hardware_signal: "GREEN" | "YELLOW" | "RED"
            reason_summary: explanatory string
        """
        defects_count = len(defects)

        # Rule 1: Immediate Reject on Spoilage (Grade C)
        if grade == "C":
            return (
                "FAIL",
                "RED",
                "Grade C: Kondisi mata/insang mengindikasikan ikan tidak segar (reject)."
            )

        # Rule 2: Reject on Physical Defects / Contamination
        if defects_count > 0:
            defect_labels = ", ".join(sorted(list(set(d.get("label", "defek") for d in defects))))
            return (
                "FAIL",
                "RED",
                f"Terdeteksi {defects_count} kecacatan fisik/kontaminasi ({defect_labels})."
            )

        # Rule 3: Secondary Inspection on Ambiguous Grade B
        if grade == "B" and grade_confidence < confidence_threshold:
            return (
                "CONDITIONAL",
                "YELLOW",
                f"Grade B dengan tingkat keyakinan moderat ({int(grade_confidence * 100)}% < {int(confidence_threshold * 100)}%). Butuh konfirmasi visual operator."
            )

        # Rule 4: Quality Passed
        if grade in ["A", "B"]:
            return (
                "PASS",
                "GREEN",
                f"Kualitas ikan memenuhi standar kelayakan ekspor (Grade {grade}, {int(grade_confidence * 100)}% keyakinan)."
            )

        return (
            "FAIL",
            "RED",
            "Kondisi ikan tidak memenuhi standar mutu minimum."
        )
