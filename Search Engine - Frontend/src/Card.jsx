import React, { useEffect } from 'react';

const Card = ({docData,searchId}) => {

  const monthData={
    '1':'Jan',
    '2':'Fed',
    '3':'Mar',
    '4':'Apr',
    '5':'May',
    '6':'Jun',
    '7':'Jul',
    '8':'Aug',
    '9':'Sep',
    '10':'Oct',
    '11':'Nov',
    '12':'Dec'
  }
   const month = monthData[docData.Month]

  return (
    <div  className="w-full mt-4 mb-4 p-6 bg-white rounded-md shadow-md">
      <div className="flex">
        {/* Left Partition */}
        <div className="w-1/2 pr-4">
          <h2 className="text-xl font-bold mb-4">{docData.documentTitle}</h2>
          <h2 className="text-xl font-bold mb-4">{docData.year}</h2>
          <h2 className="text-xl font-bold mb-4">{month}</h2>
        </div>

        {/* Right Partition */}
        <div className="w-1/2 pl-4">
          <h2 className="text-xl font-bold mb-4">{docData.Authors}</h2>
          <a href={docData.link} target='_blank'>Click here to read</a>

        </div>
      </div>
    </div>
  );
};

export default Card;
