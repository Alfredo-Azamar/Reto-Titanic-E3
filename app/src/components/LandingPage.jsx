import { useNavigate } from "react-router-dom";
import "../styles/landingpage.css";
import logoWhite from "../assets/logoWhite.png";
import colorscircle from "../assets/colorscircle.png";
import React, { useState, useEffect } from "react";
import openingGif from "../assets/gifLoaded.gif";

const LandingPage = (Props) => {
  const [showLogIn, setShowLogIn] = useState(false);
  const [gifLoaded, setGifLoaded] = useState(false);

  // Setting timeout to show the login form.
  useEffect(() => {
    setTimeout(() => {
      setShowLogIn(true);
    }, 2000);
  }, []);

  // Initial gif loading - opening for login.
  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      setGifLoaded(true);
    };
    img.src = openingGif;
  }, []);

  const navigate = useNavigate();

  const handleFillFormClick = () => {
    navigate("/quoter");
  };

  return (
    <div className="landing-page">
      <img
        src={openingGif}
        alt="Log In Gif"
        className={`background-gif ${gifLoaded ? "visible" : ""}`}
      />
      <div
        className={`background-overlay ${
          showLogIn && gifLoaded ? "blurred" : ""
        }`}
      />
      <div className={`landing-content ${showLogIn && gifLoaded ? "visible" : ""}`}>
        <header className="landing-header">
          <div>
            <img src={logoWhite} alt="Logo" className="logo" />
          </div>
          <div className="menu-icon">☰</div>
        </header>

        <div className="landing-container">
          <div>
            <div className="landing-text">
              <h1>Stellar Exodus</h1>
              <h2>
                <span>Explore. </span>
                <span className="highlight">Travel. </span>
                Live.
              </h2>
            </div>

            <p className="landing-subtext">
              Connect with a universe of possibilities. Are you ready for the
              adventure?
            </p>

            <div className="button-container">
              <button
                onClick={handleFillFormClick}
                className="fill-form-button"
              >
                Fill form
              </button>
            </div>
          </div>

          <div>
            <img
              src={colorscircle}
              alt="Colors Circle"
              className="colorscircle"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;
