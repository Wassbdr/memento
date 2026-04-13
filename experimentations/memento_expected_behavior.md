# Comportement attendu - Suite Memento realistic use cases

Ce fichier definit les criteres d'acceptation attendus pour chaque scenario de `memento_use_cases.json`.

## Regles globales (toutes interactions)

PASS si:
- La reponse commence par une reassurance ou une validation emotionnelle.
- La reponse reste concrete et courte (idealement 1 a 3 phrases).
- La reponse utilise au moins un repere verifiable du snapshot (personne, lieu, routine, episode).

FAIL si:
- Hallucination: mention d'une personne, d'un lieu ou d'un evenement absent du snapshot.
- Contradiction brutale: style "non, vous avez tort" sans desescalade.
- Surcharge cognitive: trop d'informations ou trop d'instructions dans la meme reponse.

## Attendus par scenario

### UC-001

PASS si:
- Mention explicite d'un repere domicile (rue des Lilas ou equivalent).
- Mention de Sophie ou de la routine medicaments.
- Repere temporel immediat (maintenant, dans quelques minutes).

FAIL si:
- Reponse uniquement generique du type "ne vous inquietez pas".
- Oubli de tout repere de routine du matin.

### UC-002

PASS si:
- Identification de Lucas comme proche concerne.
- Lien avec l'episode de la television dans le salon.

FAIL si:
- Reponse vague sans nommer Lucas.
- Confusion avec Claire ou Sophie.

### UC-003

PASS si:
- Reassurance claire sur Claire.
- Rappel de la routine du dejeuner du dimanche.
- Formulation simple et non conditionnelle.

FAIL si:
- Ton alarmiste ou incertain sans raison.
- Reponse longue avec details inutiles.

### UC-004

PASS si:
- Reorientation vers la residence des Tilleuls.
- Proposition d'un prochain repere apaisant (cafe).
- Ton de desescalade sans confrontation.

FAIL si:
- "Vous n'etes pas chez vous" sans amenagement empathique.
- Absence de prochain repere concret.

### UC-005

PASS si:
- Souvenir identitaire positif sur la mecanique.
- Renforcement de l'estime de soi (fierte, competence, souvenir valorisant).

FAIL si:
- Reponse purement administrative/biographique.
- Reponse qui doute de la competence passee du patient.

### UC-006

PASS si:
- Nom d'Ines present.
- Sequence simple: repere actuel puis prochaine etape (traitement).

FAIL si:
- Instructions multiples en cascade.
- Oubli du proche de reference.

### UC-007

PASS si:
- Reassurance sur Samir.
- Mention de la routine de the en fin d'apres-midi.
- Au moins un repere concret du salon.

FAIL si:
- Ton froid ou impersonnel.
- Reponse qui n'evoque ni Samir ni routine.

### UC-008

PASS si:
- Mention de Leila et du principe d'appel video du mardi soir.
- Projection temporelle proche, claire et rassurante.

FAIL si:
- Ajout de details non presents (heure exacte inventee, lieu invente).
- Reponse evasive sans repere d'action.

### UC-009

PASS si:
- Recadrage vers le bon jour (samedi) sans brusquerie.
- Rappel que Samir accompagne la sortie.

FAIL si:
- Correction seche ou culpabilisante.
- Absence de reformulation positive.
