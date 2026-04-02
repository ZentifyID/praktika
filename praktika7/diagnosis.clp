(deftemplate symptom
  (slot name)
  (slot value))

(deffacts initial-state
  (symptom (name problem) (value unknown)))

(defrule power-supply-problem
  (symptom (name power) (value no))
  (symptom (name beeps) (value none))
  =>
  (printout t "Power supply issue." crlf)
  (assert (diagnosis power-supply)))

(defrule memory-or-video-problem
  (symptom (name beeps) (value yes))
  =>
  (printout t "Problem with RAM or video card." crlf)
  (assert (diagnosis hardware-issue)))

(defrule video-card-problem
  (symptom (name screen) (value black))
  (symptom (name beeps) (value yes))
  =>
  (printout t "Video card issue." crlf)
  (assert (diagnosis video-card)))

(defrule boot-device-problem
  (symptom (name error-message) (value "No boot device"))
  =>
  (printout t "Hard drive or connection problem." crlf)
  (assert (diagnosis boot-device)))

(defrule overheating-problem
  (symptom (name reboot) (value spontaneous))
  =>
  (printout t "CPU overheating or unstable power supply." crlf)
  (assert (diagnosis overheating)))

(defrule end-diagnosis
  ?f <- (diagnosis ?d)
  =>
  (printout t "Diagnosis complete. Recommendation: contact a service center." crlf)
  (retract ?f))