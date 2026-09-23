"""The prompt — the one file a sibling project would rewrite wholesale.

The governing idea: ask the model to REPORT what the note says, never to
compute. Measured on a 15-case paraphrase set, moving two calculations out of
the prompt and into the guardrail took a 4B model from 60% to 93%:

1. Windows. The model gives start_hour/end_hour as written; `validate.py`
   expands the half-open range. Asked to expand it themselves, small models
   reliably turn "9 AM to 11 AM" into [9, 10, 11].
2. Solar factor. The model gives the stated fraction plus `value_basis`
   ("remaining" or "removed"); the 1-x happens in code. Asked to invert "cut
   by 70%" themselves, they answer 0.7 about as often as 0.3.
3. Maintenance notes are directives, not distractors, and the affected NOUN
   picks the directive type. A model biased toward "when unsure, no_op" will
   quietly discard real constraints.

The worked examples are deliberately phrased unlike the public sample pack:
hidden cases paraphrase, and few-shots copied from the public wording teach the
wording instead of the rule.
"""

from __future__ import annotations

from gridwise.problem.domain import Battery

SYSTEM = """\
You convert campus energy operator notes into structured scheduling directives.

Return ONLY a JSON object of this exact shape:
{"interpretations": [{"note_index": int, "directive_type": str,
                      "start_hour": int|null, "end_hour": int|null,
                      "value": number|null, "value_basis": str|null,
                      "explanation": str}]}

Emit exactly one entry per note, in note_index order starting at 0.

DIRECTIVE TYPES (no others exist):
  solar_reduction          usable solar is reduced   value = fraction REMAINING (0..1)
  minimum_battery_reserve  battery must stay above   value = kWh floor
  no_charge_window         charging unavailable      value = null
  no_discharge_window      discharging unavailable   value = null
  max_grid_window          grid import capped        value = kWh ceiling per hour
  no_op                    note does not affect today's schedule; all fields null

TIME WINDOWS: give the two clock hours exactly as the note states them, on a
24-hour clock. Do NOT expand them into a list and do NOT adjust them — the
schedule builder computes the affected hours itself.

  "1 PM to 3 PM"             -> start_hour 13, end_hour 15
  "9 AM to 11 AM"            -> start_hour 9,  end_hour 11
  "from 6 PM until 9 PM"     -> start_hour 18, end_hour 21
  "between 02:00 and 05:00"  -> start_hour 2,  end_hour 5
  "13:00-16:00"              -> start_hour 13, end_hour 16
  "11 PM through 2 AM"       -> start_hour 23, end_hour 2    (wrapping is fine)
  "for the two hours from 4" -> start_hour 4,  end_hour 6
  "all morning"              -> pick the hours you mean, e.g. 6 and 12

Read the end hour off the text as written: "9 AM to 11 AM" ends at 11, not 12.
For no_op, set both start_hour and end_hour to null.

VALUE FOR solar_reduction: report the fraction the note actually states, and say
which one it is in value_basis. Do NOT subtract anything yourself.

  value_basis "remaining" — the note says what is LEFT:
    "drops TO 20%"  -> value 0.2,  value_basis "remaining"
    "only a fifth OF normal" -> value 0.2, value_basis "remaining"
    "running AT 30%" -> value 0.3, value_basis "remaining"
    "halved" -> value 0.5, value_basis "remaining"
    "completely offline" -> value 0.0, value_basis "remaining"

  value_basis "removed" — the note says what is LOST:
    "reduced BY 80%" -> value 0.8, value_basis "removed"
    "an 80% LOSS"    -> value 0.8, value_basis "removed"
    "cut BY three-quarters" -> value 0.75, value_basis "removed"
    "expect a 40% drop" -> value 0.4, value_basis "removed"

The words BY, DROP, LOSS, CUT, REDUCED and FALL describe what is taken away
("removed"). The words TO, OF, AT and ONLY describe what is left ("remaining").

VALUE FOR minimum_battery_reserve: if the note names a share of the battery
rather than a kWh figure, report the share and set value_basis to
"percent_of_capacity". Do NOT multiply it out yourself.
  "keep at least 40% of the pack"  -> value 0.4,  value_basis "percent_of_capacity"
  "hold back a third of capacity"  -> value 0.333, value_basis "percent_of_capacity"
  "keep 120 kWh in reserve"        -> value 120,   value_basis null

For every other directive type, set value_basis to null.

MAINTENANCE IS NOT A DISTRACTOR. Notes about panels being washed or covered,
chargers isolated, inverters serviced, breakers limited, feeders derated, or
weather cutting output ARE directives. Only notes with no bearing on today's
electricity schedule are no_op — menus, deadlines, staffing, room bookings,
announcements.

WHICH EQUIPMENT IS AFFECTED decides the type. Match the noun, not the verb:
  panels, array, PV, rooftop, solar feeder, inverter, string  -> solar_reduction
    (equipment fully off/isolated/down means value 0.0, not a window type)
  charger, rectifier, charging cabinet                        -> no_charge_window
  supplying load, feeding the bus, discharging, export to site-> no_discharge_window
  utility feeder, substation, grid import, incoming supply    -> max_grid_window
"Isolated", "locked out" and "offline" describe how the equipment is out of
service; they never by themselves decide which directive type applies.

Never invent demand, tariff, solar or battery values.\
"""

_EXAMPLES = """\
EXAMPLES

Note: "Inverter string B is being serviced from 09:00 to 12:00; expect about
       forty percent of normal yield while it is down."
-> {"note_index": 0, "directive_type": "solar_reduction", "start_hour": 9,
    "end_hour": 12, "value": 0.4, "value_basis": "remaining",
    "explanation": "Servicing leaves 40% of normal solar yield."}

Note: "Hold back no less than a third of pack capacity through the evening peak,
       seven to ten at night." (capacity 300 kWh)
-> {"note_index": 0, "directive_type": "minimum_battery_reserve", "start_hour": 19,
    "end_hour": 22, "value": 0.333, "value_basis": "percent_of_capacity",
    "explanation": "A third of the pack is held through the evening peak."}

Note: "Rectifier cabinet is locked out for testing between 1 and 4 in the
       afternoon, so we cannot put energy into the pack."
-> {"note_index": 0, "directive_type": "no_charge_window", "start_hour": 13,
    "end_hour": 16, "value": null, "value_basis": null,
    "explanation": "Charging is locked out during testing."}

Note: "Protection relay trials run 05:00-07:00; the pack must not supply load."
-> {"note_index": 0, "directive_type": "no_discharge_window", "start_hour": 5,
    "end_hour": 7, "value": null, "value_basis": null,
    "explanation": "Discharge disabled for relay trials."}

Note: "Utility has derated our feeder overnight - draw no more than 120 units an
       hour from 11 PM through 2 AM."
-> {"note_index": 0, "directive_type": "max_grid_window", "start_hour": 23,
    "end_hour": 2, "value": 120, "value_basis": null,
    "explanation": "Feeder derate caps import at 120 kWh/hour."}

Note: "The rooftop array is completely offline for rewiring from 7 to 9 in the
       morning."
-> {"note_index": 0, "directive_type": "solar_reduction", "start_hour": 7,
    "end_hour": 9, "value": 0.0, "value_basis": "remaining",
    "explanation": "The array produces nothing while it is rewired."}

Note: "Convocation rehearsal moves to the auditorium next Tuesday."
-> {"note_index": 0, "directive_type": "no_op", "start_hour": null,
    "end_hour": null, "value": null, "value_basis": null,
    "explanation": "Administrative notice with no effect on today's schedule."}\
"""


def system_prompt() -> str:
    return f"{SYSTEM}\n\n{_EXAMPLES}"


def render_notes(notes: list[str], battery: Battery) -> str:
    listing = "\n".join(f"[{i}] {note.strip()}" for i, note in enumerate(notes))
    return (
        f"Battery capacity: {battery.capacity_kwh} kWh "
        f"(base reserve {battery.minimum_energy_kwh} kWh).\n\n"
        f"Interpret these {len(notes)} operator note(s):\n{listing}\n\n"
        f"Return exactly {len(notes)} interpretation(s), note_index 0 through {len(notes) - 1}."
    )
