# Sécurité

## Signaler une vulnérabilité

Ne publiez pas de vulnérabilité dans une *issue* publique. Utilisez la fonction **« Report a vulnerability »** de l'onglet *Security* du dépôt GitHub (signalement privé). Nous accusons réception sous 5 jours ouvrés.

## Périmètre

Krito exécute localement un modèle ONNX sur du texte fourni par l'appelant. Points d'attention :
- **Chargez uniquement des modèles de confiance.** Un fichier ONNX ou un dépôt Hugging Face malveillant peut exploiter une faille du runtime.
- **Limitez la taille des requêtes** au niveau de votre API. Krito tronque les textes à 512 tokens, mais la tokenisation d'un texte énorme consomme quand même des ressources.
- **Les textes classés peuvent contenir des données personnelles.** Krito ne les journalise pas ; si vous le faites, appliquez vos règles RGPD.
