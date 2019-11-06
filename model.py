from sklearn import tree

can_throw_out = 0
cannot_throw_out = 1

def class_to_text(throwout_value):
    if (throwout_value == 0): 
        return 'Can throw out'
    if (throwout_value == 1):
        return 'Cannot throw out'
    else: return "Unknown value"

strings = {
    'Beholdere og dunke fra opvaske- og mildere rengøringsmidler, vaske- og skyllemidler, kattebakker, udtjente opvaskebaljer og æsker/beholdere til opbevaring af mad': 1.0,
    'Plastikbakker og poser fra kød, fisk eller grønsager, drikkedunke og plastflasker uden pant': 2.0,
    'Værktøjskasse udelukkende lavet af plast': 3.0,
    'Grydeskeer lavet af plast': 4.0,
    'Serverings- og opbevaringsskåle lavet af plast': 5.0,
    'Plastposer også de små fra fx slik': 6.0,
    'Indpakning af plast fra fx afskårne blomster eller fødevarer': 7.0,
    'Bobleplast': 8.0,
    'Plastfolier': 9.0,
    'Bestik lavet af plast': 10.0,
    'Emballage, der har indeholdt kemikalier': 11.0,
    'Maling og andet farligt affald som fx printerpatroner og makeup': 12.0,
    'Uhygiejniske ting som fx tandbørsten og opvaskebørsten': 13.0,
    'Ringbind af plast - ældre ringbind kan være af PVC, og skal afleveres på genbrugsstationen. Nye ringbind skal i dagrenovationsbeholderen, da det er et for sammensat produkt af plast metal og pap': 14.0,
    'PVC, fx tagrender og gummistøvler skal afleveres på genbrugsstationen': 15.0,
    'Affald, som er sammensat af flere forskellige materialer, der ikke er til at skille ad, skal i beholderen til restaffald': 16.0
}

training = [
    [strings['Beholdere og dunke fra opvaske- og mildere rengøringsmidler, vaske- og skyllemidler, kattebakker, udtjente opvaskebaljer og æsker/beholdere til opbevaring af mad']], 
    [strings['Plastikbakker og poser fra kød, fisk eller grønsager, drikkedunke og plastflasker uden pant']],
    [strings['Værktøjskasse udelukkende lavet af plast']],
    [strings['Grydeskeer lavet af plast']],
    [strings['Serverings- og opbevaringsskåle lavet af plast']],
    [strings['Plastposer også de små fra fx slik']],
    [strings['Indpakning af plast fra fx afskårne blomster eller fødevarer']],
    [strings['Bobleplast']],
    [strings['Plastfolier']],
    [strings['Bestik lavet af plast']],
    [strings['Emballage, der har indeholdt kemikalier']],
    [strings['Maling og andet farligt affald som fx printerpatroner og makeup']],
    [strings['Uhygiejniske ting som fx tandbørsten og opvaskebørsten']],
    [strings['Ringbind af plast - ældre ringbind kan være af PVC, og skal afleveres på genbrugsstationen. Nye ringbind skal i dagrenovationsbeholderen, da det er et for sammensat produkt af plast metal og pap']],
    [strings['PVC, fx tagrender og gummistøvler skal afleveres på genbrugsstationen']],
    [strings['Affald, som er sammensat af flere forskellige materialer, der ikke er til at skille ad, skal i beholderen til restaffald']]
]

classes = [
    can_throw_out, can_throw_out, can_throw_out, can_throw_out, can_throw_out, can_throw_out, can_throw_out, can_throw_out, can_throw_out, can_throw_out, 
    cannot_throw_out, cannot_throw_out, cannot_throw_out, cannot_throw_out, cannot_throw_out, cannot_throw_out
]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(training, classes)
predict_result = clf.predict([[strings['Maling og andet farligt affald som fx printerpatroner og makeup']]])
print(class_to_text(predict_result))
tree.plot_tree(clf)