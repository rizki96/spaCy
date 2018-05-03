# encoding: utf8
from __future__ import unicode_literals

from ..symbols import *

"""
TAG_MAP = {
	"ADV":      {POS: ADV},
	"NOUN":     {POS: NOUN},
	"ADP":      {POS: ADP},
	"PRON":     {POS: PRON},
	"SCONJ":    {POS: SCONJ},
	"PROPN":    {POS: PROPN},
	"DET":      {POS: DET},
	"SYM":      {POS: SYM},
	"INTJ":     {POS: INTJ},
	"PUNCT":    {POS: PUNCT},
	"NUM":      {POS: NUM},
	"AUX":      {POS: AUX},
	"X":        {POS: X},
	"CONJ":     {POS: CONJ},
	"ADJ":      {POS: ADJ},
	"VERB":     {POS: VERB}
}
"""

TAG_MAP = {
	".": 	{POS: PUNCT},
	",": 	{POS: PUNCT},
	"--": 	{POS: PUNCT},
	":": 	{POS: PUNCT},
	";": 	{POS: PUNCT},
	"-": 	{POS: PUNCT},
	"\"": 	{POS: PUNCT},
	"...": 	{POS: PUNCT},
	"GM": 	{POS: PUNCT},
	"OP": 	{POS: PUNCT},
	"CP": 	{POS: PUNCT},
	"CC": 	{POS: CONJ},
	"SC": 	{POS: SCONJ},
	"CDP": 	{POS: NUM},
	"CDI": 	{POS: NUM},
	"CDO": 	{POS: NUM},
	"CDC": 	{POS: NUM},
	"DT": 	{POS: DET},
	"FW": 	{POS: X},
	"IN": 	{POS: ADP},
	"JJ": 	{POS: ADJ},
	"NN": 	{POS: NOUN},
	"NNP": 	{POS: NOUN},
	"NNG": 	{POS: NOUN},
	"PRP": 	{POS: PRON},
	"PRN": 	{POS: PRON},
	"PRL": 	{POS: PRON},
	"WP": 	{POS: PRON},
	"RB": 	{POS: ADV},
	"RP": 	{POS: PART},
	"NEG": 	{POS: PART},
	"SYM": 	{POS: SYM},
	"UH": 	{POS: INTJ},
	"VB": 	{POS: VERB},
	"VBT": 	{POS: VERB},
	"VBI": 	{POS: VERB},
	"MD": 	{POS: VERB}
}
