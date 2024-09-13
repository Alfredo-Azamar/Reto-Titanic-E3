import React from "react";
import "../styles/modal.css";
import { useState, useEffect } from "react";

const Modal = ({ acceptance, onClose, destin }) => {
  const [planetD, setPlanetD] = useState(null);

  useEffect(() => {
    if (destin === 0) {
        setPlanetD("TRAPPIST-1e");
      } else if (destin === 1) {
        setPlanetD("55 Cancri e");
      } else {
        setPlanetD("PSO J318.5-22");
      }
  }, [destin]);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <h2>Travel Quotation Result</h2>
        <p>
          {acceptance === "1"
            ? `Congratulations! Your trip to ${planetD} has been accepted!`
            : "Sorry, your trip was not accepted :(."}
        </p>
        <button onClick={onClose} className="close-button">
          Close
        </button>
      </div>
    </div>
  );
};

export default Modal;
