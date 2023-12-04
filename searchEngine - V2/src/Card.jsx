import React, { useEffect } from 'react';

const Card = ({docData,searchId}) => {

  return (
    <div  className=" mt-4 mb-4 p-6 bg-white rounded-md shadow-md">
      <div className="flex">
        {/* Left Partition */}
        <div className="w-1/2 pr-4">
          <h2 className="text-xl font-bold mb-4">{docData.documentTitle}</h2>
          <h2 className="text-xl font-bold mb-4">{docData.year}</h2>
        </div>

        {/* Right Partition */}
        <div className="w-1/2 pl-4">
          <h2 className="text-xl font-bold mb-4">{searchId}</h2>
          <a href={docData.link} target='_blank'>Doc Link</a>
        </div>
      </div>
    </div>
  );
};

export default Card;
